# packet_buddy
pcap analysis provided by chatGPT 4 Turbo

## Getting started

Clone the repo

Make a .env file inside the packet_buddy folder (where the packet_buddy.py file is located)

put this in the file:
```console
OPENAI_API_KEY="<your openapi api key>"
```

## Bring up the server
docker-compose up 

## Visit localhost
http://localhost:8505

### Usage
This has been tested with a variety of small .pcap files and works best with smaller data sets. If possible use wireshark filters or other methods to limit the size of the .pcap and number of packets you wish to 'chat' with.
