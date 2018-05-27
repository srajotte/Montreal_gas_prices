#!python3

import datetime
import os, shutil

import numpy as np
from scipy.interpolate import interp1d
from scipy import fftpack
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
import cv2

DAY_NAMES= ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday","Sunday"]

def parse_range_line(line):
	return line.replace(' ','').replace('\n','').split(';')

def parse_range(filename):
	with open(filename) as f:
		date_line = next(f)
		price_line = next(f)

	# Each tick on the time scale is the end of the day, or the beginning of the next day.
	# By moving all the data by 12 hours, the data is placed midday instead of at the edge between two days.
	# This should reduce the errors when the datetimes are converted to days.
	date_range = [datetime.datetime.strptime(x, "%Y-%m-%d") + datetime.timedelta(days=0.5) for x in parse_range_line(date_line)]
	price_range = [float(x) for x in parse_range_line(price_line)]
	return date_range, price_range

def interpolate_data(x, y, x_interpolated, kind='cubic'):
	interpolator = interp1d(x, y, kind=kind, fill_value="extrapolate")
	return interpolator(x_interpolated)

def days_in_range(date_range):
	return (date_range[1] - date_range[0]).days

def extract_price_curve(img, date_range, price_range, DAYS_CALIB_OFFSET=0.0, DAYS_CALIB_RANGE_OFFSET=0.0):
	price_pixel_count = img.shape[0]
	day_pixel_count = img.shape[1]

	price_linspace = np.linspace(price_range[0], price_range[1], num=price_pixel_count, endpoint=True)
	days = days_in_range(date_range)
	days_linspace = np.linspace(0, days + DAYS_CALIB_RANGE_OFFSET, num=day_pixel_count, endpoint=True)
	days_linspace = days_linspace + DAYS_CALIB_OFFSET

	data = img[:,:,0]
	DATA_THRESHOLD = 128
	COLUMN_SHAPE = data.shape[0]
	daily_average = np.empty(day_pixel_count)
	daily_min = np.empty(day_pixel_count)
	daily_max = np.empty(day_pixel_count)
	for i in range(day_pixel_count):
		column = data[:,i]
		indices = COLUMN_SHAPE - 1 - np.where(column < DATA_THRESHOLD)[0]
		values = price_linspace[indices]
		daily_average[i] = np.mean(values)
		sorted_values = np.sort(values)
		N = len(sorted_values)
		if N >= 3:
			daily_min[i] = sorted_values.take(1, mode='clip') # second smallest
			daily_max[i] = sorted_values.take(len(sorted_values)-2, mode='clip') # second largest
		else:
			daily_min[i] = sorted_values[0] # smallest value
			daily_max[i] = sorted_values[-1] # largest value
	return days_linspace, daily_average, daily_min, daily_max

def extract_daily_values(img, date_range, price_range, DAYS_CALIB_OFFSET, DAYS_CALIB_RANGE_OFFSET):
	days, daily_average, daily_min, daily_max = extract_price_curve(img, date_range, price_range, DAYS_CALIB_OFFSET, DAYS_CALIB_RANGE_OFFSET)	
	daily_day_offsets, daily_prices_peak_pinned, daily_min_days, daily_min_prices, daily_max_days, daily_max_prices = daily_values(date_range, days, daily_average, daily_min, daily_max)
	return daily_day_offsets, daily_prices_peak_pinned, daily_min_days, daily_min_prices, daily_max_days, daily_max_prices

def interpolate_price_curve(date_range, points_per_day, days, daily_average):
	days_count = days_in_range(date_range)
	days_interpolated = np.linspace(0, days_count, num=days_count*points_per_day, endpoint=False)
	daily_average_interpolated = interpolate_data(days, daily_average, days_interpolated, kind='quadratic')
	return days_interpolated, daily_average_interpolated

def day_offsets_to_dates(start_date, offsets):
	offsets_float = offsets.astype('float')
	return [start_date + datetime.timedelta(days=x) for x in offsets_float]

def day_of_week(dates, offset=datetime.timedelta(hours=0)):
	return np.array([(x+offset).weekday() for x in dates], dtype='int') #- datetime.timedelta(hours=12)

def compute_weekday_distribution(weekdays):
	hist_values, _ = np.histogram(weekdays, bins=7, range=(0,6))
	dist = {}
	for name, value in zip(DAY_NAMES, hist_values):
		dist[name] = value
	return dist

def plot_histogram(x, y, label, color, hatch_pattern):
	bar_plot_kwargs = {"linewidth":1.0}
	edge_color = matplotlib.colors.colorConverter.to_rgba(color, alpha=1.0)
	fill_color = matplotlib.colors.colorConverter.to_rgba(color, alpha=0.25)
	plt.bar(x, y, label=label, color=fill_color, edgecolor=edge_color, hatch=hatch_pattern, **bar_plot_kwargs)

def get_price(day_offset_querries, day_offsets, prices):
	return np.interp(day_offset_querries, day_offsets, prices, left=np.nan, right=np.nan)

def pin_peaks(prices, min_days, min_prices, max_days, max_prices):
	min_days = set(np.round(min_days).astype('int'))
	max_days = set(np.round(max_days).astype('int'))
	pinned_prices = prices.copy()
	for i, p in enumerate(prices):
		if i in max_days:
			pinned_prices[i] = max_prices[i]
		elif i in min_days:
			pinned_prices[i] = min_prices[i]
	return pinned_prices

def find_peaks(date_range, days, daily_average, daily_min, daily_max):
	INTERPOLATION_POINTS = 10
	days_interpolated, daily_average_interpolated = interpolate_price_curve(date_range, INTERPOLATION_POINTS, days, daily_average)
	_, daily_min_interpolated = interpolate_price_curve(date_range, INTERPOLATION_POINTS, days, daily_min)
	_, daily_max_interpolated = interpolate_price_curve(date_range, INTERPOLATION_POINTS, days, daily_max)

	MIN_MAX_ORDER = 9
	min_indices = scipy.signal.argrelextrema(daily_min_interpolated, np.less, order=MIN_MAX_ORDER)[0] 
	max_indices = scipy.signal.argrelextrema(daily_max_interpolated, np.greater, order=MIN_MAX_ORDER)[0]

	min_days_offsets = days_interpolated[min_indices]
	min_prices = daily_min_interpolated[min_indices]
	max_days_offsets = days_interpolated[max_indices]
	max_prices = daily_max_interpolated[max_indices]
	return min_days_offsets, min_prices, max_days_offsets, max_prices

def daily_values(date_range, days, daily_average, daily_min, daily_max):
	daily_day_offsets, daily_prices = interpolate_price_curve(date_range, 1, days, daily_average)
	daily_datetimes = day_offsets_to_dates(date_range[0], daily_day_offsets)
	daily_dayofweek = day_of_week(daily_datetimes)

	min_days, min_prices, max_days, max_prices = find_peaks(date_range, days, daily_average, daily_min, daily_max)

	_, daily_prices_min = interpolate_price_curve(date_range, 1, days, daily_min)
	_, daily_prices_max = interpolate_price_curve(date_range, 1, days, daily_max)
	daily_prices_peak_pinned = pin_peaks(daily_prices, min_days, daily_prices_min, max_days, daily_prices_max)
	
	daily_min_days = np.round(min_days).astype('int')
	daily_max_days = np.round(max_days).astype('int')

	daily_min_prices = get_price(daily_min_days, daily_day_offsets, daily_prices_peak_pinned)
	daily_max_prices = get_price(daily_max_days, daily_day_offsets, daily_prices_peak_pinned)

	return daily_day_offsets, daily_prices_peak_pinned, daily_min_days, daily_min_prices, daily_max_days, daily_max_prices

def match_price_peaks(daily_min_days, daily_max_days, daily_prices):
	valid_daily_max_days = daily_max_days[1:-1]
	valid_daily_max_prices = daily_prices[valid_daily_max_days]
	previous_days = valid_daily_max_days-1
	previous_min_days = np.zeros(np.shape(valid_daily_max_days), dtype='int')
	for i in range(len(valid_daily_max_prices)):
		previous_min_days[i] = daily_min_days[np.argmin(np.abs(daily_min_days - valid_daily_max_days[i]))]
	previous_min_prices = daily_prices[previous_min_days]
	price_increases = valid_daily_max_prices - previous_min_prices

	increases_dict = {
		'days' : np.array([previous_min_days, valid_daily_max_days], dtype='int'),
		'prices' : np.array([previous_min_prices, valid_daily_max_prices])
	}
	return increases_dict

def dir_exists(dir):
	return os.path.exists(dir) and os.path.isdir(dir)

def clear_dir(dir):
	if not dir_exists(dir) : return
	for entry in os.listdir(dir):
		e_path = os.path.join(dir, entry)
		if os.path.isdir(e_path):
			os.rmdir(e_path)
		else:
			os.remove(e_path)

def test_create_dir(dir):
	if not dir_exists(dir):
		os.mkdir(dir)

def savefig(filename, destination_dir):
	plt.savefig(os.path.join(destination_dir, filename + '.png'), dpi=200)

def analysis_period_spectrum(daily_day_offsets, daily_prices_peak_pinned):
	n = len(daily_prices_peak_pinned)
	price_fft = np.fft.fft(daily_prices_peak_pinned - np.mean(daily_prices_peak_pinned)) / n
	time_step = daily_day_offsets[1] - daily_day_offsets[0]
	sample_freq = fftpack.fftfreq(price_fft.size, d=time_step)
	power = np.abs(price_fft)

	data_range = np.array(range(n//2), dtype='int')
	data_range = data_range[sample_freq[data_range] != 0.0]
	period_data = 1.0/sample_freq[data_range] 
	power_data = power[data_range]

	period_max_indices = scipy.signal.argrelextrema(power_data, np.greater, order=3)[0]
	power_max_values = power_data[period_max_indices]
	period_max_values = period_data[period_max_indices]
	weekly_peak_index = np.argmin(np.abs(period_max_values - 7.0))
	weekly_peak_index = period_max_indices[weekly_peak_index]
	weekly_peak_period = period_data[weekly_peak_index]
	weekly_peak_power = power_data[weekly_peak_index]
	return period_data, power_data, weekly_peak_period, weekly_peak_power

def analysis_increase_period(daily_min_days, daily_max_days, daily_prices_peak_pinned):
	increases_dict = match_price_peaks(daily_min_days, daily_max_days, daily_prices_peak_pinned)
	increases_prices = increases_dict['prices']
	increases_days = increases_dict['days']
	price_increases = increases_prices[1] - increases_prices[0]
	
	print("Average price increase : ", np.mean(price_increases))
	print("Minimum price increase : ", np.min(price_increases))
	print("Maximum price increase : ", np.max(price_increases))

	increase_period = increases_days[1][1:] - increases_days[1][:-1]
	return increase_period

def analysis_weekday_peaks_distribution(date_range, daily_min_days, daily_max_days):
	min_weekdays = day_of_week(day_offsets_to_dates(date_range[0], daily_min_days))
	max_weekdays = day_of_week(day_offsets_to_dates(date_range[0], daily_max_days))

	min_weekdays_distr = compute_weekday_distribution(min_weekdays)
	max_weekdays_distr = compute_weekday_distribution(max_weekdays)
	return min_weekdays_distr, max_weekdays_distr

def analysis_peak_diff_evolution(daily_day_offsets, daily_prices_peak_pinned, daily_max_days, FUTURE_DAYS):
	valid_max_days = daily_max_days[(daily_max_days > 0) & (daily_max_days <= np.max(daily_day_offsets) - (FUTURE_DAYS-1))]
	price_diff = np.zeros((len(valid_max_days), FUTURE_DAYS))
	for i in range(FUTURE_DAYS):
		price_diff[:,i] = get_price(valid_max_days + i, daily_day_offsets, daily_prices_peak_pinned)
	price_diff = (price_diff.T - price_diff[:,0]).T
	return price_diff

def keep_fullweeks(date_range, daily_day_offsets, daily_prices_peak_pinned):
	daily_datetimes = day_offsets_to_dates(date_range[0], daily_day_offsets)
	daily_dayofweek = day_of_week(daily_datetimes)	

	first_monday_index = np.where(daily_dayofweek == 0)[0][0]
	last_sunday_index = np.where(daily_dayofweek == 6)[0][-1]

	daily_fullweek_indices = np.arange(first_monday_index, last_sunday_index+1)
	daily_fullweek_dayofweek = daily_dayofweek[daily_fullweek_indices]
	daily_fullweek_prices = daily_prices_peak_pinned[daily_fullweek_indices]
	return daily_fullweek_dayofweek, daily_fullweek_prices

def analysis_weekday_average_price(date_range, daily_day_offsets, daily_prices_peak_pinned):
	daily_fullweek_dayofweek, daily_fullweek_prices = keep_fullweeks(date_range, daily_day_offsets, daily_prices_peak_pinned)

	DAYS_IN_WEEK = 7
	dayofweek_average = np.zeros(DAYS_IN_WEEK, dtype='float')
	for i in range(DAYS_IN_WEEK):
		dayofweek_indices = daily_fullweek_dayofweek==i
		dayofweek_average[i] = np.mean(daily_fullweek_prices[dayofweek_indices])
	annual_average = np.mean(daily_fullweek_prices)
	return dayofweek_average, annual_average

def main():
	FIGURES_DIR = './figures'
	test_create_dir(FIGURES_DIR)
	clear_dir(FIGURES_DIR) 

	img_filename = './data/montreal 2018-05-25/montreal_12 months_binary.png'
	range_filename = './data/montreal 2018-05-25/montreal_12 months_range.txt'

	date_range, price_range = parse_range(range_filename)
	img = cv2.imread(img_filename)

	DAYS_CALIB_OFFSET = 0.0
	DAYS_CALIB_RANGE_OFFSET = 0.0 
	
	daily_day_offsets, daily_prices_peak_pinned, daily_min_days, daily_min_prices, daily_max_days, daily_max_prices = extract_daily_values(img, date_range, price_range, DAYS_CALIB_OFFSET, DAYS_CALIB_RANGE_OFFSET)	

	#
	# Daily values.
	#
	plt.figure()
	plt.title('Daily gasoline price in Montreal')
	plt.plot(daily_day_offsets, daily_prices_peak_pinned, alpha=0.75)
	plt.plot(daily_min_days, daily_min_prices, '.', label='min', color='black')
	plt.plot(daily_max_days, daily_max_prices, '.', label='max', color='red')
	#plt.plot(increases_days, increases_prices)
	plt.gca().grid(True, alpha=0.5)
	plt.xlabel('Days since %s' %str(date_range[0].date()))
	plt.ylabel('Price [cents/L]')
	plt.tight_layout()
	savefig('annual daily prices', FIGURES_DIR)

	#
	# Period spectrum.
	#
	period_data, power_data, weekly_peak_period, weekly_peak_power = analysis_period_spectrum(daily_day_offsets, daily_prices_peak_pinned)

	plt.figure()
	plt.semilogx(period_data, power_data)
	plt.plot(weekly_peak_period, weekly_peak_power, '.', color='red')
	plt.gca().annotate('%.2f days' % np.around(weekly_peak_period, 2), xy=(weekly_peak_period, weekly_peak_power), xytext=(3.0, 1.75),
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            )
	plt.title('Annual price period spectrum')
	plt.xlabel('Period [days]')
	plt.ylabel('Power')
	plt.gca().grid(True, alpha=0.5)
	plt.tight_layout()
	savefig('period spectrum', FIGURES_DIR)

	#
	# Price increase period.
	#
	increase_period = analysis_increase_period(daily_min_days, daily_max_days, daily_prices_peak_pinned)
	increase_period_min = np.min(increase_period)
	increase_period_max = np.max(increase_period)

	plt.figure()
	bins = np.arange(increase_period_max-increase_period_min+2)-0.5+np.min(increase_period)
	plt.hist(increase_period, bins=bins, color='green', alpha=0.75, histtype='bar', ec='black')
	plt.gca().grid(True, alpha=0.5)
	plt.xticks(np.arange(increase_period_min, increase_period_max+1, step=1.0))
	plt.title('Distribution of time period between two consecutive price increases.')
	plt.xlabel('Period [days]')
	plt.ylabel('Count')
	plt.tight_layout()
	savefig('price increase period distribution', FIGURES_DIR)

	#
	# Price peaks day of week distribution.
	#
	min_weekdays_distr, max_weekdays_distr = analysis_weekday_peaks_distribution(date_range, daily_min_days, daily_max_days)
	
	plt.figure()
	plot_histogram(min_weekdays_distr.keys(), min_weekdays_distr.values(), label='min', color='green', hatch_pattern='//')
	plot_histogram(max_weekdays_distr.keys(), max_weekdays_distr.values(), label='max', color='red', hatch_pattern='\\\\')
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
	plt.ylabel('Count')
	plt.legend()
	plt.title('Distribution of price peaks by day of week.')
	plt.tight_layout()
	savefig('price peak weekday distribution', FIGURES_DIR)

	#
	# Price difference evolution after increase.
	#
	FUTURE_DAYS = 14 + 1
	price_diff = analysis_peak_diff_evolution(daily_day_offsets, daily_prices_peak_pinned, daily_max_days, FUTURE_DAYS)
	days_from_peak = np.arange(FUTURE_DAYS)
	
	plt.figure()
	for i in range(price_diff.shape[0]):
		plt.plot(days_from_peak, price_diff[i,:], '.-', alpha=0.75)
	plt.gca().grid(True, alpha=0.5)
	plt.title('Evolution of daily gasoline prices after a price increase')
	plt.xlabel('Days since price increase')
	plt.ylabel('Price difference [cents/L]')
	plt.tight_layout()
	savefig('price evolution after increase', FIGURES_DIR)

	mean_savings = np.mean(price_diff, axis=0)
	savings_stddev = np.std(price_diff, axis=0)

	N_STDDEV = 2
	price_max_stddev = mean_savings + N_STDDEV*savings_stddev
	price_min_stddev = mean_savings - N_STDDEV*savings_stddev

	plt.figure()	
	stddev_kwargs = {'color' : 'red', 'linewidth' : 0.5}
	plt.plot(days_from_peak, price_max_stddev, **stddev_kwargs)
	plt.plot(days_from_peak, price_min_stddev, **stddev_kwargs)
	plt.gca().fill_between(days_from_peak, price_max_stddev, price_min_stddev, color='red', alpha=0.25)
	plt.plot(days_from_peak, mean_savings, '.-', linewidth=2.0, color='black')
	plt.grid(True, alpha=0.5)
	plt.title('Evolution of average daily gasoline prices after a price increase')
	plt.xlabel('Days since price increase')
	plt.ylabel('Price difference [cents/L]')
	plt.tight_layout()
	savefig('average price evolution after increase', FIGURES_DIR)

	#
	# Average price per day of week.
	#
	dayofweek_average, annual_average = analysis_weekday_average_price(date_range, daily_day_offsets, daily_prices_peak_pinned)
	print("annual average : ", annual_average)

	plt.figure()
	plt.gca().yaxis.grid(True, alpha=0.5)
	plot_histogram(DAY_NAMES, dayofweek_average, label=None, color='green', hatch_pattern=None)
	plt.plot([-0.5,6.5],[annual_average,annual_average], '-', color='red', label='annual average')
	plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
	plt.title('Average gasoline prices by day of week')
	plt.ylabel('Price [cents/L]')
	plt.legend()
	plt.tight_layout()
	savefig('weekday average price', FIGURES_DIR)

	dayofweek_annual_mean_diff = dayofweek_average - annual_average

	plt.figure()
	plt.gca().yaxis.grid(True, alpha=0.5)
	plot_histogram(DAY_NAMES, dayofweek_annual_mean_diff, label=None, color='green', hatch_pattern=None)
	plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
	plt.title('Price difference by day of week compared to annual average')
	plt.ylabel('Price difference [cents/L]')
	plt.tight_layout()
	savefig('weekday annual average difference', FIGURES_DIR)

	#plt.show()

if __name__ == '__main__':
	main()