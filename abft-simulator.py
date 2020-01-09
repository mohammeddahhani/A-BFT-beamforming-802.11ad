# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import namedtuple,defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("N", help="Number of stations" ,type=int)
parser.add_argument("NS", help="Number of slots in a period" ,type=int)
parser.add_argument("MAX_ATTEMPT", help="Max. number of attempts" ,type=int)
parser.add_argument("MAX_IDLE", help="Idle backoff window size" ,type=int)
parser.add_argument("--time", help="Simulation time", type=int)
parser.add_argument("--tofile", help="Write stats to file",action="store_true")
if len(sys.argv)< 5:
    parser.print_help(sys.stderr)
    sys.exit(1)
args = parser.parse_args()

STA = args.N	
if args.time > 0:
	MAX_PERIOD = args.time
else:
	MAX_PERIOD = 10000
NS 			= args.NS
MAX_SLOTS 	= MAX_PERIOD*NS
MAX_SLOTS 	= MAX_PERIOD*NS
MAX_ATTEMPT = args.MAX_ATTEMPT	# allowed number of successive failed RSS attempts
MAX_IDLE 	= args.MAX_IDLE		# max number of periods a station stays in IDLE

log_coll 	= np.zeros(MAX_SLOTS,dtype=int)
log_idle 	= np.zeros(MAX_SLOTS,dtype=int)
log_tx 	 	= np.zeros(MAX_SLOTS,dtype=int)
log_th 	 	= np.zeros(MAX_SLOTS,dtype=int)
log_retx 	= np.zeros(MAX_SLOTS,dtype=int)
log_max_retx= np.zeros(MAX_PERIOD,dtype=int)

per_period_success 			= []
per_period_cond_idle_count 	= []
echec_par_trame 			= []

log_periods_toidle	= [] 
log_periods_tosucc	= []  
log_slots_tosucc	= []  
log_retx_tosucc		= [] 
log_slots   		= []  
log_active_sta	 	= []

pkt_id  = 0	# ongoing RSS id
s_index = 0 # current slot

"""
	Station model:

	A station performs one Responder Sector Sweep (RSS) at a time by uniformly choosing
	a number in [0, MAX_ATTEMPT-1] in each period.
	If the station fails MAX_ATTEMPT consecutive times the same RSS it becomes idle.
	An idle station backoff a number of periods chosen uniformly [0, MAX_IDLE-1].

	- id: station's id
	- retx_id: id of the ongoing failed RSS attempt
	- is_idle: 1 if the station is idle, 0 otherwise.
	- idle_log: the period in whichs the station became idle (0 if never been idle)
"""
Station = namedtuple('Sta',['id','retx_pid','is_idle', 'idle_log']) 

"""
	RSS attempt:

	We model a RSS as one packet transmission.
	We abstract the feedback mechanism and instead keep count of the number
	of failed attempt to determine the RSS attempt success.

	- id: RSS id
	- sta: station performing the RSS
	- slot: slot index in which the RSS started
	- retx_count: number of time the ongoing RSS failed
	- retx_log: number of periods since the ongoing RSS started
"""
Packet  = namedtuple('Pkt',['id','sta','slot','retx_count','retx_log'])

timeslots = defaultdict(list)
stations  = defaultdict(list)

# Initialize stations
for i in xrange(STA):
	stations[i]=Station(i,0,0,0)


##################################################
#                                                #
#                  AUX. FUNCTIONS                #
#                                                #
##################################################

def to_file(file, STA, NS, succ_delay, succ_count, idle_proba, succ_proba, succ_repart, law_I):
	BASE_DIR 		= "./"
	file 			= "{}simu_{}_slots_{}_maxAttempt_{}_maxIdle".format(BASE_DIR, NS, MAX_ATTEMPT, MAX_IDLE)
	serialized_arr	= ''
	for x in law_I:
		serialized_arr += '\t{}'.format(x)
	with open(file, "a+") as myfile:
	    myfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}{7}\n"\
	    	.format(NS, STA, succ_delay, succ_count, idle_proba, succ_proba, succ_repart, serialized_arr))

"""
	Program a transmission 
"""
def add_to_slot(pkt, slot):
	global timeslots
	timeslots[slot].append(pkt)

def get_period_boundaries(slot):
	i= 0
	if slot % NS == 0: 
		res = (slot, slot + NS -1)
	else:
		while (slot + i) % NS != 0:
			i+=1
		res = ((slot+i)-NS, slot+i-1)
	return res

"""
	reset station 
"""
def reset_station(sid):
	global stations
	stations[sid] = Station(sid,0,0,0)
	return sid

"""
	Create a paquet with new index
	Clean station retransmission history
	Transmit paquet
"""
def new_attempt_uniform(sta, pid, slot):
	global stations
	pkt = None
	boff = 0
	sid,_,_,_ = stations[sta]
	
	pkt = Packet(pid + 1, sid, slot, 0, 0)
	reset_station(pkt.sta)
	
	boff = np.random.randint(0,NS)
	add_to_slot(pkt, slot+boff)

	return boff,pkt


def get_per_period_proba(arr1, arr2):
	assert(len(arr1) == len(arr2))
	_probas = []

	for i in range(len(arr2)):
		_tmp = 0
		if arr1[i] !=0:
			_tmp = (1.0*arr2[i])/arr1[i]
		_probas.append(_tmp)

	return np.mean(_probas)


##################################################
#                                                #
#                  SIMULATOR LOGIC               #
#                                                #
##################################################

def start_period(pid, slot):
	_active_stas = 0
	boff = 0
	pkt = None
	p_count = 0
	first_slot_tx = [] 	# contains new RSS attempts and previously failed RSS attempts.
	attempting_again= []  # contains failed RSS attempts in prev period that will 
						  # attempt again if retx count permits it.
	tx = len(timeslots[slot]) # number of contending stations in current slot.
	(start,_) = get_period_boundaries(slot)
	period = start/NS

	if tx == 0: # all active stations in previous period succeeded their attempts
		for sta in xrange(STA):
			if stations[sta].retx_pid == 0 and stations[sta].is_idle == 0:
				boff, new_pkt = new_attempt_uniform(sta, pid, slot) # uniform backoff			
				if boff == 0: first_slot_tx.append(new_pkt)
				_active_stas += 1
				pid = new_pkt.id
		
				
	elif tx > 0: # Active stations from previous periods that failed their attempt.
		for pkt in timeslots[slot]: 
			sid = pkt.sta
			attempting_again.append(sid)

			boff = np.random.randint(0, NS)
			if boff == 0: first_slot_tx.append(pkt)
			else: add_to_slot(pkt, boff+slot) 			

			_active_stas +=1	

		for sid in xrange(STA): # Active Stations that succeeded their attempts in the past period
			if (sid not in attempting_again) and (stations[sid].is_idle == 0): # sta not in idle state
				boff, new_pkt = new_attempt_uniform(sid, pid, slot)
				if boff == 0: first_slot_tx.append(new_pkt)
				pid = new_pkt.id

				_active_stas +=1

	log_active_sta.append(_active_stas)

	# Solve contention in the first slot if any
	handle_slot(slot, first_slot_tx, True)

	return pid,  _active_stas

def handle_slot(slot, contending, first_slot=False):
	tx = len(contending)
	(period,_) = get_period_boundaries(slot)
	period = period/NS 

	if tx > 1: # number of contending stations
		log_coll[slot] += 1
		log_tx[slot]   += tx
		log_retx[slot] += tx

		for pkt in contending:
			boff = np.random.randint(0, NS)
			(start, end) = get_period_boundaries(slot)

			pkt = pkt._replace(retx_count = pkt.retx_count + 1); # update failure count		
			pkt = pkt._replace(retx_log = pkt.retx_log + 1);	 # update failure history
			pid, sid, _, _, _ = pkt

			if pkt.retx_count < MAX_ATTEMPT:
				if slot+boff+1 > end: 
					(start, end) = get_period_boundaries(slot+boff+1)
				else:
					start = slot+1+boff
				add_to_slot(pkt, start)
			
			# Sta becomes IDLE
			else: 
				"""
					Logging stats
				"""
				log_max_retx[period] += 1
				last_period 	= stations[sid].idle_log
				if last_period == 0:
					(slot_,_) 	= get_period_boundaries(pkt.slot)
					last_period = slot_/NS
				log_periods_toidle.append(period - last_period + 1)


				boff 			= np.random.randint(0, MAX_IDLE) # idle backoff
				(start, end)	= get_period_boundaries(NS*(boff+1) + slot)			

				stations[sid] 	= Station(sid, pid, 1, start/NS)			
				pkt 			= pkt._replace(retx_count=0);
				pkt 			= pkt._replace(retx_log=pkt.retx_log);
				add_to_slot(pkt, start)	

	elif tx == 1:		
		reset_station(contending[0].sta)
		"""
			Logging stats
		"""
		(s, _) = get_period_boundaries(slot)
		log_slots.append(slot-s)

		(f_start, _) = get_period_boundaries(contending[0].slot)
		(l_start, _) = get_period_boundaries(slot)

		slots 	= slot - contending[0].slot  + 1
		periods = (l_start - f_start)/NS + 1

		log_periods_tosucc.append(periods)
		log_slots_tosucc.append(slots)
		log_retx_tosucc.append(contending[0].retx_log)

		log_th[slot] += 1
		log_tx[slot] += 1

	elif tx == 0:
		log_idle[slot] += 1


##################################################
#                                                #
#                  MAIN PROGRAM                  #
#                                                #
##################################################

print
print('Total stations:    {0}  '.format(STA))
print('Slots per period:  {0}  '.format(NS))
print('Max. attempt:      {0}  '.format(MAX_ATTEMPT))
print('Max. idle:         {0}  '.format(MAX_IDLE))
print('Time (periods):    {0} ({1} slots)'.format(MAX_SLOTS/NS, MAX_SLOTS))

  
prev_succ	= 0
succes 		= 0
idle_count 	= 0
period 		= 0

while s_index < MAX_SLOTS:
	(start, end) 	= get_period_boundaries(s_index)
	period 			= start/NS

	# run protocol
	if s_index == start:
		pkt_id, _  = start_period(pkt_id, s_index)
	else:
		handle_slot(s_index, timeslots[s_index])

	# Stats from prev period
	if s_index > 0 and s_index % NS == 0 : # begining of period and not first period
		cond_idle_count = log_max_retx[period-1]
		per_period_cond_idle_count.append(cond_idle_count)

		success= np.sum(log_th[(s_index-NS):s_index])
		per_period_success.append(success)
		
		coll= np.sum(log_coll[(s_index-NS):s_index])
		
		tx= np.sum(log_tx[(s_index-NS):s_index])

	s_index += 1


##################################################
#                                                #
#            Probability Distribution            #
#                                                #
##################################################

# Distribution of success within a period
uniq = np.unique(log_slots).tolist()
avg  = 0.0
for x in uniq:
	avg += (x)*log_slots.count(x)*1.0/len(log_slots)
succ_repart = avg

# Law of number of period before first success
uniq = np.unique(log_periods_tosucc).tolist()
avg = 0
for x in uniq:
	avg += (x-1)*log_periods_tosucc.count(x)*1.0/len(log_periods_tosucc)

# Law of number of period before a station is idle ()
uniq = np.unique(log_periods_toidle).tolist()
avg  = 0.0
law_I = np.zeros(MAX_ATTEMPT, dtype=float)
for x in uniq:
	law_I[x-1] = log_periods_toidle.count(x)*1.0/len(log_periods_toidle)

# Law of number of success in a period
uniq = np.unique(per_period_success).tolist()
avg  = 0.0
for x in uniq:
	avg += (x)*per_period_success.count(x)*1.0/len(per_period_success)
succ_count = avg
print


##################################################
#                                                #
#                  Simu Stats                    #
#                                                #
##################################################

print 					  	
print "Per period stats:\n*****************"
print
log_th		 	= np.array([np.sum(x) for x in log_th.reshape(MAX_PERIOD,NS)])
log_tx 			= np.array([np.sum(x) for x in log_tx.reshape(MAX_PERIOD,NS)])
log_coll 		= np.array([np.sum(x) for x in np.array(log_coll).reshape(MAX_PERIOD,NS)])
log_idle 		= np.array([np.sum(x) for x in np.array(log_idle).reshape(MAX_PERIOD,NS)])
channel_input 	= pkt_id*1.0 / MAX_PERIOD   
throughput		= np.mean(log_th)  			 
traffic 		= np.mean(log_tx)	#*1.0/s_index
collisions 		= np.mean(log_coll)			
idle_slots 		= np.mean(log_idle)			
retx 			= np.sum(np.array(log_retx_tosucc))*1.0/MAX_PERIOD#/_o_len

print('[+] Channel input: {0:.2f} per period - {1}'.format(channel_input, 	pkt_id))
print('[+] Throughput:    {0:.2f} per period - {1}'.format(throughput, 		np.sum(log_th)))
print('[+] Traffic:       {0:.2f} per period - {1}'.format(traffic, 		np.sum(log_tx)))
print('[+] Collisions:    {0:.2f} per period - {1}'.format(collisions,  np.sum(log_coll)))
print('[+] Empty slots:   {0:.2f} per period - {1}'.format(idle_slots,  np.sum(log_idle)))
print('[+] avg # retx:    {1:.2f} retx - {2}'.format(NS, retx,	np.sum(log_retx_tosucc)))

print 				  
print "Global stats:\n*************"
print
cond_idle_proba 	= np.mean(per_period_cond_idle_count)/STA 	# conditional proability: P['idle'/'active']. 
idle_proba 		= np.mean([1.0-(1.0*x/STA) for x in log_active_sta]) # P['idle'/'active'] + P['idle'/'idle']
old_succ_proba 		= throughput/traffic
succ_proba 		= get_per_period_proba(log_tx, log_th)
succ_delay 		= np.mean(log_periods_tosucc)-1
if len(log_periods_toidle) == 0: 
	log_periods_toidle = [0]
	idle_delay = 0
else:
	idle_delay = np.mean(log_periods_toidle)-1

print("[+] Avg number of success:    {0:.2f}".format(succ_count))	
print("[+] Success Rate:             {0:.2f}%".format(succ_count*100.0/STA))
print('[+] Periods until success:    {0:.2f} periods (max: {1} - min: {2})'\
					.format(succ_delay, np.max(log_periods_tosucc), np.min(log_periods_tosucc)))                                      
print('[+] Periods until idle:       {0:.2f} periods (max: {1} - min: {2})'\
					.format(idle_delay, np.max(log_periods_toidle), np.min(log_periods_toidle)))                                      
print('[+] Slots until success:      {0:.2f} slots (max: {1})'\
					.format(np.mean(log_slots_tosucc) - 1, np.max(log_slots_tosucc)))                        
print('[+] Proba of Idle:            {0:.2e}'\
					.format(idle_proba, cond_idle_proba))
print('[+] Proba of success:         {0:.2e}'\
					.format(succ_proba, old_succ_proba))
if args.tofile: 
	to_file(STA, NS, succ_delay, succ_count, idle_proba, succ_proba, succ_repart, law_I)


