from dotenv import load_dotenv

from open_the_valve.features.build_panel import run
from open_the_valve.io_utils.hydra_entrypoint import hydra_entrypoint

load_dotenv()

main = hydra_entrypoint(run)

if __name__ == "__main__":
    main()
