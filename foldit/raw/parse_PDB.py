
import re

def parse_PDB(fp):
    entries = re.findall('^IRDATA .*$', fp.read(), re.MULTILINE)
    pdb = {}
    pdb['PDL'] = []
    # print(fp.name)
    for e in entries:
        m = re.match('IRDATA (\w*) (.*)$', e)
        if m is None:
            raise AttributeError("corrupted IRDATA entry {} in {}".format(e, fp.name))
        if m.group(1) == 'PDL':
            s = m.group(2)
            header_raw = s.strip('. ').split(' |')[0].split(',')
            header = {
                'uid': header_raw[2],
                'gid': header_raw[3],
                'score': header_raw[5]
            }
            actions_raw = re.split(' \S*?\|', s.strip('. '))[1:] # sometimes random cyrillic characters show up, regex takes care of that

            actions = dict((k,int(v)) for k,v in (x.split('=') for x in actions_raw if x.startswith('Action')))
            macros = dict((k.strip('MacroScrBO'),int(v)) for k,v in (x.split('=') for x in actions_raw if x.startswith('MacroScrB')))
            pdl = {
                'header': header,
                'actions': actions,
                'macros': macros
            }
            pdb['PDL'].append(pdl)
        elif m.group(1) == 'SCORE':
            pdb['SCORE'] = float(m.group(2))
        else:
            pdb[m.group(1)] = m.group(2)

    pdb['PDL'].reverse() # pdb file has them in reverse chronological order top to bottom, so we put them in chronological order here
    return pdb

if __name__ == '__main__':
    fp = open("data/solution_2002713/top/solution_uid_0000_0000267374_0000993077_0352233051.ir_solution.pdb")
    data = parse_PDB(fp)

    
