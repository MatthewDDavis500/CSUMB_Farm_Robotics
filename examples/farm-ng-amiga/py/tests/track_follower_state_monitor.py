import asyncio
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file

async def main():
    config = proto_from_json_file(Path("track_follower_state_config.json"), EventServiceConfig())
    client = EventClient(config)

    print("Monitoring Track Follower /state ...")
    async for event, msg in client.subscribe(config.subscriptions[0], decode=True):
        print("EVENT PATH:", event.uri.path)
        print("EVENT QUERY:", event.uri.query)
        print("MSG TYPE:", type(msg))
        print("MSG:", msg)
        print("-" * 60)

asyncio.run(main())