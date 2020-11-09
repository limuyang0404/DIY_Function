# coding=UTF-8
from segy_cut import segy_cut

segy_cut(r'C:\Users\Administrator\Desktop\cb27_600-2000ms_final.sgy', ['CB27_river.dat'], slice_class='iline')
segy_cut(r'C:\Users\Administrator\Desktop\cb27_600-2000ms_final.sgy', ['CB27_river.dat'], slice_class='xline')
segy_cut(r'C:\Users\Administrator\Desktop\cb27_600-2000ms_final.sgy', ['CB27_river.dat'], slice_class='time')