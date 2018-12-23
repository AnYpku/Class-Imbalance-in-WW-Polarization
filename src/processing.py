import numpy as np
import pandas as pd

class processData():
    def __init__(self):
        self.evtcats = ['status', 'e', 'mother1', 'mother2', 'pz', 'px', 'py',
                        'm', 'color1', 'color2', 'lifetime', 'spin', 'id']
        self.partlist = ['j1', 'j2', 'W1', 'W2']
        self.cols = ['e.W1', 'e.W2', 'e.j1', 'e.j2', 'px.W1', 'py.W1', 'pz.W1',
                     'px.W2', 'py.W2', 'pz.W2', 'px.j1', 'py.j1', 'pz.j1',
                     'px.j2', 'py.j2', 'pz.j2', 'spin.W1', 'spin.W2']
        self.rawcols = [e + '.' + p for p in self.partlist for e in self.evtcats]

    def pt(self, px, py):
        # transverse momentum
        return np.sqrt(px**2 + py**2)

    def phi(self, px, py):
        # azimuthal angle
        return np.arctan2(px, py)

    def eta(self, pz, e):
        #pseudorapidity
        return np.arctanh(pz / e)

    def mm(self, e, px, py, pz):
        # invariant mass
        return np.sqrt(e**2 - (px**2 + py**2 + pz**2))

    def n_long(self, spin1, spin2):
        # returns the number of longitudinally polarized W-bosons in an event
        return 2 - (spin1**2 + spin2**2)

    def get_events(self, csvfile):
        #read from .csv file
        df = pd.read_csv(csvfile,
                         header=None,
                         names=self.rawcols,
                         usecols=self.cols)

        # target function 'n_lon' = number of longitudinally polarized W-bosons
        df['n_lon'] = self.n_long(df['spin.W1'], df['spin.W2']).astype('int')

        # pt
        # energies are given in units of GeV
        df['pT.W1'] = self.pt(df['px.W1'], df['py.W1'])
        df['pT.W2'] = self.pt(df['px.W2'], df['py.W2'])
        df['pT.j1'] = self.pt(df['px.j1'], df['py.j1'])
        df['pT.j2'] = self.pt(df['px.j2'], df['py.j2'])

        # eta
        df['eta.W1'] = self.eta(df['pz.W1'], df['e.W1'])
        df['eta.W2'] = self.eta(df['pz.W2'], df['e.W2'])
        df['eta.j1'] = self.eta(df['pz.j1'], df['e.j1'])
        df['eta.j2'] = self.eta(df['pz.j2'], df['e.j2'])

        # phi
        df['phi.W1'] = self.phi(df['px.W1'], df['py.W1'])
        df['phi.W2'] = self.phi(df['px.W2'], df['py.W2'])
        df['phi.j1'] = self.phi(df['px.j1'], df['py.j1'])
        df['phi.j2'] = self.phi(df['px.j2'], df['py.j2'])

        # derived variables help neural networks pick up the relevant
        # features faster, but do not significantly affect performance

        #VBF jets
        df['mm.jj'] = self.mm(df['e.j1'] + df['e.j2'],
                              df['px.j1'] + df['px.j2'],
                              df['py.j1'] + df['py.j2'],
                              df['pz.j1'] + df['pz.j2'])
        df['delta_eta.jj'] = np.abs(df['eta.j1'] - df['eta.j2'])
        delta_phis = np.abs(df['phi.j1'] - df['phi.j2'])
        df['delta_phi.jj'] = np.array([dp if dp <= np.pi else 2*np.pi - dp
                                       for dp in delta_phis])

        # WW system
        df['mm.WW'] = self.mm(df['e.W1'] + df['e.W2'],
                              df['px.W1'] + df['px.W2'],
                              df['py.W1'] + df['py.W2'],
                              df['pz.W1'] + df['pz.W2'])
        df['e.WW'] = df['e.W1'] + df['e.W2']
        df['pT.WW'] = self.pt(df['px.W1'] + df['px.W2'],
                              df['py.W1'] + df['py.W2'])
        df['eta.WW'] = self.eta(df['pz.W1'] + df['pz.W2'],
                                df['e.W1'] + df['e.W2'])
        df['phi.WW'] = self.phi(df['px.W1'] + df['px.W2'],
                                df['py.W1'] + df['py.W2'])

        # drop unnecessary columns
        df = df.drop(self.cols[4:], axis=1)

        # attempt to save memory by using smaller datatypes when possible
        # e.g. ~50% savings in memory with float32 vs. float64
        converted_ints = (df
                          .select_dtypes(include=['int'])
                          .apply(pd.to_numeric, downcast='unsigned'))
        converted_floats = (df
                            .select_dtypes(include=['float'])
                            .apply(pd.to_numeric, downcast='float'))
        df[converted_ints.columns] = converted_ints
        df[converted_floats.columns] = converted_floats

        # reorder columns such that n_lon is 1st
        df = df[df
                .columns[4:5]
                .append(df
                        .columns[:4])
                .append(df
                        .columns[5:])]

        return df