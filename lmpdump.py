# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import time as tm

class lmpdump():

	def __init__(self, *args, **kwargs):
		"""
		args:	args[0] = name of the dump file
		kwargs: loadmode - all: loads all snaps from the trj

		"""

		self.filename = args[0]
		self.finaldict = {}
		MAXNUMBEROFSNAPSHOTS = 10000000
		loadmode = kwargs.get('loadmode')

		if loadmode == "all":
			tmpdata = {}

			NumberOfAtomLines = self.pickNatoms()
			starttime = tm.time()
			f = open(self.filename, "r")
			for nsnap in range(MAXNUMBEROFSNAPSHOTS):
				rvalue = self.read_snapshot(f, NumberOfAtomLines)
				if rvalue == 0:
					print('Done with loading %d snapshots in %.2fs.'%(
						nsnap+1, tm.time() - starttime))
					break
				elif rvalue == 1:
					continue
			f.close()


	def read_snapshot(self, f, n):
		NofAtoms = n
		DUMPHEADER = 9

		floatentry = ["x", "y", "z", 
			"xu", "yu", "zu",
			"xs", "ys", "zs",
			"xsu", "ysu", "zsu",
			"vx", "vy", "vz",
			"fx", "fy", "fz", "c_pepa"]
		intentry = ["id", "type", "mol", "c_cn"]

		# Load chunks of the file
		chunk_size = NofAtoms+DUMPHEADER
		snap_info = [next(f, []) for x in range(chunk_size)]

		# check if snap is full or empty, 
		# if empty means dump is done
		if snap_info[1] == []:
			return 0

		# Get snap time
		ST = int(snap_info[1].split()[0])

		# Get box extremes - x, y, z
		xs = np.array(snap_info[5].split()).astype(float)
		ys = np.array(snap_info[6].split()).astype(float)
		zs = np.array(snap_info[7].split()).astype(float)
		BE = (xs[0], xs[1], ys[0], ys[1], zs[0], zs[1])

		# Get snap atom info
		SAI = {}
		Columns = snap_info[8].split()[2:]

		tmpinfo = []
		infolen = len(snap_info) - DUMPHEADER
		[tmpinfo.append(snap_info[DUMPHEADER + i].split()) for i in range(infolen)]

		floatvalues = np.array(tmpinfo)
		for index, head in enumerate(Columns):
			if head in floatentry:
				ToAdd = floatvalues[:,index].astype(float)
			elif head in intentry:
				ToAdd = floatvalues[:,index].astype(int)
			else:
				ToAdd = floatvalues[:,index]

			SAI[head] = ToAdd

		AtomInfo = pd.DataFrame(SAI)

		# Updates the overall dictionary
		# containing all important info
		self.finaldict[ST] = (BE, AtomInfo)
		return 1



	def pickNatoms(self):
		f = open(self.filename, "r")

		for index, line in enumerate(f):
			if index == 3:
				f.close()
				return int(line)


	def IndexOrder(self):
		for key in self.finaldict.keys():
			tmp = self.finaldict[key][1].sort_values(
				by="id",
				ascending=True)
			self.finaldict[key] = (self.finaldict[key][0], tmp)








