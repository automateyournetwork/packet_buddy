# packet_buddy
pcap analysis provided by chatGPT 4 Turbo

## Getting started

Clone the repo

## Bring up the server
docker-compose up 

## Visit Ollama and download your model(s)
http://localhost:3002

Gear / settings button

Models

Download phi, llama2, gemma, etc

## Start Packet Buddy
http://localhost:8505

### Usage
This has been tested with a variety of small .pcap files and works best with smaller data sets. If possible use wireshark filters or other methods to limit the size of the .pcap and number of packets you wish to 'chat' with. For larget .pcaps I would recommend Packet RAPTOR instead.

Upload your PCAP 
Pick Your Model
Ask questions about the PCAP

The tool will download the Instructor-XL model dynamically, be patient, the first time you launch it, in order to provide free open source embeddings
