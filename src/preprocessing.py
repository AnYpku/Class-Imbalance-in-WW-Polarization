import csv
import pylhe

def lhe_to_csv(importfile, exportfile, particlei=0, particlef=None):
    """
    A barebones way to extract all the information from each event from a
    .LHE file and write it to a .csv file

    I choose to not do the xml processing myself, but to use pylhe, which is
    available at https://github.com/lukasheinrich/pylhe
    Note that pylhe uses Python2, not Python3!

    if particlei or particlef are non-zero not all of the particles in the
    event will be written to file
    """
    LHEgen = pylhe.readLHE(importfile)

    with open(exportfile, 'w') as file1:
        writes = csv.writer(file1, delimiter=',')
        for event in LHEgen:
            eventlist = []
            for particle in event['particles'][particlei:particlef]:
                eventlist += particle.values()
            writes.writerow(eventlist)

if __name__ == '__main__':
    lhe_to_csv('../data/jjWpmWpm_undecayed_01.lhe',
               '../data/jjWpmWpm_undecayed_01.csv',
               particlei=-4)
    lhe_to_csv('../data/jjWpmWpm_undecayed_02.lhe',
               '../data/jjWpmWpm_undecayed_02.csv',
               particlei=-4)
    lhe_to_csv('../data/jjWpmWpm_undecayed_03.lhe',
               '../data/jjWpmWpm_undecayed_03.csv',
               particlei=-4)
