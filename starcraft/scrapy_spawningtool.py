# scrap "pro" replay files and metadata from spawningtool.com
# 7/10/2020
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
    replay_nums = sorted([r[:r.index(".")].strip(string.ascii_letters + "_") for r in
                   os.listdir("replays") if r.startswith("spawningtool") and r.endswith("_meta.html")])
    for replay_num in replay_nums:
        if os.path.exists(f"replays/spawningtool_{replay_num}_meta.json") and not reprocess:
            continue

        with open(f"replays/spawningtool_{replay_num}_meta.html", 'rb') as infile:
            tree = html.fromstring(infile.read())

        try:
            with open(f"replays/spawningtool_{replay_num}_meta.json", 'w') as json_out:
                replay_json = {"tags": {}}
                # select Players div (there should only be 1)
                players_div = tree.xpath("//h3[text()='Players']/..")
                assert len(players_div) == 1
                players_div: html.Element = players_div[0]


                # select player names
                name_elems = players_div.xpath(".//h4")
                for name_elem in name_elems:
                    player_json = {"tags": {}}
                    # assume no spaces allowed in bnet names
                    text = name_elem.text_content()
                    m = re.match(r"(\S+)(?: \((\S+)\))?( - Winner!)?", text)

                    name = m.group(2) if m.group(2) else m.group(1)
                    player_json["winner"] = True if m.group(3) else False
                    player_json["name"] = name
                    if m.group(2):
                        player_json["alias"] = m.group(1)
                    replay_json[name] = player_json
                    # select league label
                    league = name_elem.xpath(".//following-sibling::ul[1]/li/b[text()='League:']/../text()")
                    player_json["league"] = league[0].strip() if len(league) == 1 else None

                tag_divs = tree.xpath("//div[@class='tags-category-wrapper']")
                tags = {}
                for tag_div in tag_divs:
                    category = tag_div.attrib["category"]
                    tag_elems = tag_div.xpath(".//a")
                    if category == "Player":
                        pid_to_name = {tag_elem.attrib["data-pid"] : tag_elem.text for tag_elem in tag_elems}
                    else:
                        assert len(tag_elems) == 1 or all(tag_elem.attrib["data-pid"] for tag_elem in tag_elems)
                        tags[category] = {pid: [t.text for t in ts] for pid, ts in groupby(sorted(tag_elems, key=lambda t: t.attrib["data-pid"]), 
                                                                               lambda t: t.attrib["data-pid"])}

                for cat, d in tags.items():
                    for p, v in d.items():
                        if p in pid_to_name:
                            # correct for case differences
                            if pid_to_name[p] not in replay_json:
                                names = [k for k in replay_json if k != "tags"]
                                pid_to_name[p] = names[[n.lower() for n in names].index(pid_to_name[p].lower())]

                            replay_json[pid_to_name[p]]["tags"].setdefault(cat, list()).extend(v)
                        else:
                            replay_json["tags"].setdefault(cat, list()).extend(v)

                json.dump(replay_json, json_out)
        except Exception as e:
            print(f"FAILED to process meta for {replay_num}")
            print(replay_json)
            print(pid_to_name)
            print(cat, d)
            print()


if __name__ == "__main__":
    # scrap replays back to Jan. 2018 (on page 417 as of 7/10/2020)
    download_replays(1, 417)
    process_metas(True)
