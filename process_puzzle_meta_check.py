from process_puzzle_meta import process_puzzle_meta
from pattern_extraction import load_extend_data
import logging, sys
logging.getLogger().setLevel(logging.DEBUG)
sys.setrecursionlimit(10000)
process_puzzle_meta('2003287', True, 40)
soln_lookup = {}
parent_lookup = {}
child_lookup = {}
data, puz_metas = load_extend_data(['2003287'], soln_lookup, parent_lookup, child_lookup, False, 0)
