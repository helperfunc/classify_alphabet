import os

for dire in os.listdir('/Users/huixu/Documents/codelabs/alphabet2cla/data_resized/'):
	with open('/Users/huixu/Documents/codelabs/alphabet2cla/misc/labels.txt', 'a') as f:
		f.write(dire)
		f.write('\n')
		