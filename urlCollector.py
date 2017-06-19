from bs4 import BeautifulSoup

import urllib2, base64
import os
import requests
from requests.auth import HTTPBasicAuth

dataset = "WT2014"

if dataset == "TREC-8":
    SystemRankingAddress = "http://trec.nist.gov/results/trec8/trec8.results.input/index.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec8/trec8.results.input/"
elif dataset == "gov2":
    SystemRankingAddress = "http://trec.nist.gov/results/trec15/terabyte-adhoc.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec15/"
elif dataset == "WT2013":
    SystemRankingAddress = "http://trec.nist.gov/results/trec22/web.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec22/"
elif dataset == "WT2014":
    SystemRankingAddress = "http://trec.nist.gov/results/trec23/web.adhoc.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec23/"

request = urllib2.Request(SystemRankingAddress)
base64string = base64.encodestring('%s:%s' % ('tipster', 'cdroms')).replace('\n', '')
request.add_header("Authorization", "Basic %s" % base64string)
resp = urllib2.urlopen(request)
destinationAddress = "/home/nahid/PycharmProjects/parser/systemRanking/"+dataset+"/"

soup = BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))

for link in soup.find_all('a', href=True):
    address = str(link['href'])
    #print address
    if dataset == "TREC-8":
        if address.find("adhoc")>=0:
            downloadAddress = RankingBaseAddress+address
            fileName =  os.path.basename(downloadAddress)
            #request = requests.get(downloadAddress, auth=('tipster', 'cdroms'))
            #request = urllib2.Request(downloadAddress)
            #base64string = base64.encodestring('%s:%s' % ('tipster', 'cdroms')).replace('\n', '')
            #request.add_header("Authorization", "Basic %s" % base64string)

            response = urllib2.urlopen(request)
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'))
            output = open(destinationAddress+fileName, "w")
            output.write(r.content)
            output.close()
    elif dataset == "gov2":
        if address.find("terabyte") >= 0:
            print address
            downloadAddress = RankingBaseAddress + address
            fileName = os.path.basename(downloadAddress)

            response = urllib2.urlopen(request)
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'))
            output = open(destinationAddress + fileName, "w")
            output.write(r.content)
            output.close()
    elif dataset == "WT2013" or dataset == "WT2014":
        if address.find("web") >= 0:
            print address
            downloadAddress = RankingBaseAddress + address
            fileName = os.path.basename(downloadAddress)
            response = urllib2.urlopen(request)
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'))
            output = open(destinationAddress + fileName, "w")
            output.write(r.content)
            output.close()
