# scrap replay files and metadata from drop.sc
# 8/02/2020
# Carleton College
# Aaron Bauer

import subprocess
from typing import List
import json
import os
import string

from lxml import html
import pycurl
from io import BytesIO


def request(url):
    c = pycurl.Curl()
    buf = BytesIO()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, ['User-Agent: Mozilla/5.0'])
    c.setopt(pycurl.WRITEDATA, buf)
    print(c.getinfo(pycurl.EFFECTIVE_URL))
    c.perform()
    return buf.getvalue()


def download_replays(start_page, end_page):
    assert start_page > 0 and start_page < end_page + 1
    for page_num in range(start_page, end_page + 1):
        html_str = request(f"https://lotv.spawningtool.com/replays/?pro_only=on&p={page_num}")

        tree = html.fromstring(html_str)
        replay_elems: List[html.Element] = tree.xpath("//a[contains(@href, 'download')]")
        assert len(replay_elems) == 25

        for replay_elem in replay_elems:
            replay_num: str = replay_elem.attrib['href'].split('/')[1]
            subprocess.run(["wget", "-O", f"replays/spawningtool_{replay_num}.SC2Replay",
                            f"https://lotv.spawningtool.com/{replay_num}/download/"])

            with open(f"replays/spawningtool_{replay_num}_meta.html", 'wb') as out:
                out.write(request(f"https://lotv.spawningtool.com/{replay_num}/"))


def process_metas(reprocess=False):
    replay_nums = [r[:r.index(".")].strip(string.ascii_letters + "_") for r in
                   os.listdir("replays") if r.startswith("spawningtool") and r.endswith("_meta.html")]
    for replay_num in replay_nums:
        with open(f"replays/spawningtool_{replay_num}_meta.html", 'rb') as infile:
            tree = html.fromstring(infile.read())

        if os.path.exists(f"replays/spawningtool_{replay_num}_meta.json") and not reprocess:
            continue

        with open(f"replays/spawningtool_{replay_num}_meta.json", 'w') as json_out:
            replay_json = {"tags": {}}
            # select Players div (there should only be 1)
            players_div = tree.xpath("//h3[text()='Players']/..")
            assert len(players_div) == 1
            players_div: html.Element = players_div[0]

            # select player names
            name_elems: List[html.Element] = players_div.xpath(".//h4")
            for name_elem in name_elems:
                player_json = {"tags": {}}
                # assume no spaces allow in bnet names
                name = name_elem.text_content()
                player_json["winner"] = "Winner" in name
                player_json["name"] = name.split()[0]
                replay_json[name.split()[0]] = player_json


                # select league label
                league = name_elem.xpath(".//following-sibling::ul[1]/li/b[text()='League:']/../text()")
                player_json["league"] = league[0].strip() if len(league) == 1 else None

            # scrape tag info
            tag_divs = tree.xpath("//div[@class='tags-category-wrapper']")
            for tag_div in tag_divs:
                category = tag_div.attrib["category"]
                tag_elems = tag_div.xpath(".//a")
                for tag_elem in tag_elems:
                    # a player name, if present, is stored in the title attribute
                    if tag_elem.attrib["title"] in replay_json:
                        # there can be multiple tags per player per category, so store them in a list
                        replay_json[tag_elem.attrib["title"]]["tags"].setdefault(category, list()).append(tag_elem.text_content())
                    else:
                        replay_json["tags"].setdefault(category, list()).append(tag_elem.text_content())

            json.dump(replay_json, json_out)

if __name__ == "__main__":
    # scrap replays back to Dec. 2017 (id 6000000)
    download_replays(6000000, 15649277)
    process_metas(True)
