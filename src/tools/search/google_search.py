from typing import List

from dotenv import load_dotenv

load_dotenv(verbose=True)

import os
from time import sleep
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from googlesearch.user_agents import get_useragent

from src.proxy import proxy_env
from src.tools.search.base import SearchItem, WebSearchEngine


def _req(term, results, tbs, lang, start, proxies, timeout, safe, ssl_verify, region):
    params = {
        "q": term,
        "num": results + 2,  # Prevents multiple requests
        "hl": lang,
        "start": start,
        "safe": safe,
        "gl": region,
    }
    if tbs is not None:
        params["tbs"] = tbs

    resp = requests.get(
        url="https://www.google.com/search",
        headers={"User-Agent": get_useragent(), "Accept": "*/*"},
        params=params,
        proxies=proxies,
        timeout=timeout,
        verify=ssl_verify,
        cookies={
            "CONSENT": "PENDING+987",  # Bypasses the consent page
            "SOCS": "CAESHAgBEhIaAB",
        },
    )
    resp.raise_for_status()
    return resp


def google_search(
    term,
    num_results=10,
    tbs=None,
    lang="en",
    proxy=None,
    advanced=False,
    sleep_interval=0,
    timeout=5,
    safe="active",
    ssl_verify=None,
    region=None,
    start_num=0,
    unique=False,
):
    """Search the Google search engine"""

    # Proxy setup
    proxies = (
        {"https": proxy, "http": proxy}
        if proxy and (proxy.startswith("https") or proxy.startswith("http"))
        else None
    )

    start = start_num
    fetched_results = 0  # Keep track of the total fetched results
    fetched_links = set()  # to keep track of links that are already seen previously

    while fetched_results < num_results:
        # Send request
        resp = _req(
            term,
            num_results - start,
            tbs,
            lang,
            start,
            proxies,
            timeout,
            safe,
            ssl_verify,
            region,
        )

        # put in file - comment for debugging purpose
        # with open('google.html', 'w') as f:
        #     f.write(resp.text)

        # Parse
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", class_="ezO2md")
        new_results = 0  # Keep track of new results in this iteration

        for result in result_block:
            # Find the link tag within the result block
            link_tag = result.find("a", href=True)
            # Find the title tag within the link tag
            title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
            # Find the description tag within the result block
            description_tag = result.find("span", class_="FrIlee")

            # Check if all necessary tags are found
            if link_tag and title_tag and description_tag:
                # Extract and decode the link URL
                link = (
                    unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                    if link_tag
                    else ""
                )
            # Extract and decode the link URL
            link = (
                unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                if link_tag
                else ""
            )
            # Check if the link has already been fetched and if unique results are required
            if link in fetched_links and unique:
                continue  # Skip this result if the link is not unique
            # Add the link to the set of fetched links
            fetched_links.add(link)
            # Extract the title text
            title = title_tag.text if title_tag else ""
            # Extract the description text
            description = description_tag.text if description_tag else ""
            # Increment the count of fetched results
            fetched_results += 1
            # Increment the count of new results in this iteration
            new_results += 1
            # Yield the result based on the advanced flag
            if advanced:
                yield SearchItem(
                    title=title,
                    url=link,
                    date=None,
                    position=None,
                    source=None,
                    description=description,
                )
            else:
                yield link  # Yield only the link

            if fetched_results >= num_results:
                break  # Stop if we have fetched the desired number of results

        if new_results == 0:
            # If you want to have printed to your screen that the desired amount of queries can not been fulfilled, uncomment the line below:
            # print(f"Only {fetched_results} results found for query requiring {num_results} results. Moving on to the next query.")
            break  # Break the loop if no new results were found in this iteration

        start += 10  # Prepare for the next set of results
        sleep(sleep_interval)


def search(params):
    """
    Mock function to simulate Google search results.
    In a real-world scenario, this would interface with the Google Search API.
    """

    base_url = os.getenv("SKYWORK_GOOGLE_SEARCH_API", None)

    query = params.get("q", "")
    filter_year = params.get("filter_year", None)

    # Use local google search api
    if base_url is not None:
        with proxy_env():
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                items = response.json()
            else:
                raise ValueError(response.json())

            if "organic" not in items.keys():
                if filter_year is not None:
                    raise Exception(
                        f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                    )
                else:
                    raise Exception(
                        f"No results found for query: '{query}'. Use a less restrictive query."
                    )

            results = []
            if "organic" in items:
                for idx, page in enumerate(items["organic"]):
                    title = page.get("title", f"Google Result {idx + 1}")
                    url = page.get("link", "")
                    position = page.get("position", idx + 1)
                    description = page.get("snippet", None)
                    date = page.get("date", None)
                    source = page.get("source", None)

                    results.append(
                        SearchItem(
                            title=title,
                            url=url,
                            date=date,
                            position=position,
                            source=source,
                            description=description,
                        )
                    )
            return results

    else:  # Use remote google search api
        response = google_search(
            term=params["q"],
            num_results=params["num"],
            tbs=params.get("tbs", None),
            lang="en",
            proxy=None,
            advanced=True,
            sleep_interval=0,
            timeout=5,
        )

        results = []
        for item in response:
            results.append(item)

        return results


class GoogleSearchEngine(WebSearchEngine):
    async def perform_search(
        self,
        query: str,
        num_results: int = 10,
        filter_year: int = None,
        *args,
        **kwargs,
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        params = {
            "q": query,
            "num": num_results,
        }
        if filter_year is not None:
            params["tbs"] = (
                f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
            )

        results = search(params)

        return results
