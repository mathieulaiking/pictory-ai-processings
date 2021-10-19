from comm.messaging_queues import start_queue_listening
import sys
import os
from dotenv import load_dotenv


if __name__ == '__main__':
    try:
        load_dotenv()
        start_queue_listening()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
