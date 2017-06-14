from bs4 import BeautifulSoup
s = """<!DOCTYPE lewis SYSTEM "lewis.dtd">
 <DOC>
<DOCNO>FT911-1</DOCNO>
<PROFILE>_AN-BENBQAD8FT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / (CORRECTED) Jubilee of a jet that did what it was designed
to do
</HEADLINE>
<TEXT>
Correction (published 16th May 1991) appended to this article.
'FRANK, it flies]' shouted someone at Sir Frank Whittle during the maiden
flight of a British jet. 'Of course it does,' replied Sir Frank, who
patented the first aircraft gas turbine. 'That's what it was bloody well
designed to do, wasn't it?'
Exactly 50 years ago yesterday, the first British jet made a brief 17-minute
flight from RAF Cranwell in Lincolnshire. To celebrate the event, Mr Eric
'Winkle' Brown, a 72-year-old test pilot of the prototype Gloster Whittle
jet, Mr Geoffrey Bone, a 73-year-old engineer, and Mr Charles McClure, a
75-year-old pilot, returned to RAF Cranwell. They are seen in front of a
restored Meteor NF 11. Sir Frank was unable to attend because of ill-health.
The Gloster Whittle was not the first jet to fly: a Heinkel 178 had its
maiden flight in August 1939, 21 months before the British aircraft.
Correction (published 16th May 1991).
THE PICTURE of a Gloster Whittle jet on Page 7 of the issue of Tuesday May
14, was taken at Bournemouth Airport and not at RAF Cranwell as stated in
the caption.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
London Page 7 Photograph (Omitted).
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-2</DOCNO>
<PROFILE>_AN-BENBQABQFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / (CORRECTED) UK Company News: Geevor merger hits rocks over
pre-conditions
</HEADLINE>
<BYLINE>
   By KENNETH GOODING, Mining Correspondent
</BYLINE>
<TEXT>
Correction (published 16th May 1991) appended to this article.
Geevor, the UK mining group which has been fighting for survival since the
Canadian Imperial Bank of Commerce called in a Pounds 2.1m loan in
extraordinary circumstances in January, has suffered another set-back.
Its proposed merger with European Mining Finance, a Luxembourg-quoted
investment company, has run into problems and will not go ahead on the terms
announced in March.
Two of the pre-conditions for the merger have not been satisfied - the
raising of bank finance for the enlarged group and the termination of the
management agreement between EMF and its manager, Lion Mining Finance.
However, Geevor said it remained in talks with EMF and other parties 'which
may result in modified proposals'.
Monarch Resources, the UK-quoted mining group with operations in Venezuela,
has appointed Mr Anthony Ciali as president and chief executive officer.
This follows the recent boardroom shake-up which resulted in the departure
of seven directors and the arrival of Mr Michael Beckett as chairman.
Mr Beckett was managing director of Consolidated Gold Fields and Mr Ciali
was once a vice-president of Gold Fields Mining Corp, a US subsidiary.
Correction (published 16th May 1991).
Lion Mining Finance is not the manager of European Mining Finance as we
reported on May 14.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
London Page 22
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-3</DOCNO>
<PROFILE>_AN-BEOA7AAIFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / International Company News: Contigas plans DM900m east
German project
</HEADLINE>
<BYLINE>
   By DAVID GOODHART
</BYLINE>
<DATELINE>
   BONN
</DATELINE>
<TEXT>
CONTIGAS, the German gas group 81 per cent owned by the utility Bayernwerk,
said yesterday that it intends to invest DM900m (Dollars 522m) in the next
four years to build a new gas distribution system in the east German state
of Thuringia.
Reporting on its results for 1989-1990 the company said that the dividend
would remain unchanged at DM8.
Sales rose 9.4 per cent to DM3.37bn, but post-tax profit fell slightly from
DM31.3m to DM30.7m.
In the first half of the current year sales rose 23 per cent.
Mr Jurgen Weber, currently vice-chairman of Lufthansa, the German airline,
is today expected to be named as the successor to the chairman Mr Heinz
Ruhnau who retires at the end of 1992.
Mr Weber is currently the technical director on the Lufthansa board.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
International Page 20
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-4</DOCNO>
<PROFILE>_AN-BEOA7AAHFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / World News in Brief: Population warning
</HEADLINE>
<TEXT>
The world's population is growing faster than predicted and will consume at
an unprecedented rate the natural resources required for human survival, a
UN report said.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
International Page 1
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-5</DOCNO>
<PROFILE>_AN-BEOA7AAGFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / World News in Brief: Newspaper pays up
</HEADLINE>
<TEXT>
A Malaysian English-language newspaper agreed to pay former Singapore prime
minister Lee Kuan Yew Dollars 100,000 over allegations of corruption.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
International Page 1
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-6</DOCNO>
<PROFILE>_AN-BEOA7AAFFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / World News in Brief: Khmer Rouge snub
</HEADLINE>
<TEXT>
Khmer Rouge guerrillas refused a UN observer team's request for a visit to
the front line to observe the Cambodian ceasefire.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
International Page 1
</PAGE>
</DOC>
<DOC>
<DOCNO>FT911-7</DOCNO>
<PROFILE>_AN-BEOA7AAEFT</PROFILE>
<DATE>910514
</DATE>
<HEADLINE>
FT  14 MAY 91 / World News in Brief: Cocaine ring broken
</HEADLINE>
<TEXT>
Spanish police said they had broken up a cocaine-smuggling ring, arresting
15 Chileans and seizing Pta92m (Dollars 900,000) in cash.
</TEXT>
<PUB>The Financial Times
</PUB>
<PAGE>
International Page 1
</PAGE>
</DOC>"""
soup = BeautifulSoup(s,'html.parser')
print soup.find_all('docno')[0].next
print soup.find_all('headline')[0].next
print soup.find_all('text')[0].next
