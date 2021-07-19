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
import sc2reader


def request(url: str) -> bytes:
    """

    :param url:
    :return:
    """
    c = pycurl.Curl()
    buf = BytesIO()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, ['User-Agent: Mozilla/5.0'])
    c.setopt(pycurl.WRITEDATA, buf)
    c.perform()
    return buf.getvalue()


def download_replays(start_replay_num: int, end_replay_num: int) -> None:
    """

    :param start_replay_num:
    :param end_replay_num:
    """
    assert start_replay_num > 0 and start_replay_num < end_replay_num + 1

    official_maps = [x.strip() for x in open("valid_maps.txt").readlines()]

    for replay_num in range(start_replay_num, end_replay_num + 1):
        print()
        # running into issues with 301 redirects, probably blacklisting, so download via tor
        sp = subprocess.run(["torsocks", "wget", "--no-check-certificate", "-O", f"replays/dropsc_{replay_num}.SC2Replay",
                             f"https://sc2replaystats.com/download/{replay_num}"],
                            stderr=subprocess.DEVNULL)

        if sp.returncode == 0:
            print(f"wget {replay_num} success")
        else:
            print(f"wget {replay_num} FAILED")
            continue

        try:
            r = sc2reader.load_replay(f"replays/dropsc_{replay_num}.SC2Replay")
            if r.map_name not in official_maps or r.real_length.total_seconds() < 300 or len(r.humans) - len(r.observers) != 2:
                print(f"{replay_num} uninteresting replay ({r.map_name}, {r.real_length.total_seconds()}, {len(r.humans) - len(r.observers)}), deleting")
                os.remove(f"replays/dropsc_{replay_num}.SC2Replay")
                continue
        except Exception as e:
            print(f"{replay_num} invalid replay ({e}), deleting")
            try:
                os.remove(f"replays/dropsc_{replay_num}.SC2Replay")
                continue
            except:
                continue

        print(f"{replay_num} valid, parsing metadata")
        html_str = request(f"https://drop.sc/replay/{replay_num}")
        tree = html.fromstring(html_str)

        player_divs = tree.xpath("//div[contains(@class, 'tab-pane fade active')]/div/div")[::2]
        try:
            with open(f"replays/dropsc_{replay_num}_meta.json", 'w') as json_out:
                replay_json = {}
                for pdiv in player_divs:
                    name = pdiv.xpath(".//h3")[0].text_content()
                    info = pdiv.xpath(".//div[@class='col-md-4']")[0].text_content()
                    info = info.replace("Battle Net Profile", "").split("\n")
                    assert len(info) == 4
                    replay_json[name] = {
                        "clan": info[1].split(":")[1].strip(),
                        "name": name,
                        "apm": int(info[2].split(":")[1]),
                        "mmr": int(info[3].split(":")[1]),
                        "rank": pdiv.xpath(".//img/@src")[0].split("/")[-1][:-4],  # get string form of rank from image name used for rank insignia
                        "bnet_url": pdiv.xpath(".//a/@href")[0]
                    }
                json.dump(replay_json, json_out)
        except:
            print(f"problem with {replay_num} meta, deleting")
            os.remove(f"replays/dropsc_{replay_num}.SC2Replay")
            os.remove(f"replays/dropsc_{replay_num}_meta.json")

if __name__ == "__main__":
    # scrap replays back to Dec. 2017 (id 6000000)
    download_replays(6000000, 15649277)

