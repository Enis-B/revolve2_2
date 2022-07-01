from ast import Num, arguments
import logging
import argparse

import isaacgym
import torch

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from jlo.RL_direct_tree.rl_agent import make_agent
from jlo.RL_direct_tree.rl_optimizer import RLOptimizer
from random import Random
from jlo.RL_direct_tree.config import NUM_PARALLEL_AGENT

async def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_checkpoint",
        action="store_true",
        help="Resumes training from past checkpoint if True.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize the simulation if True.",
    )
    args = parser.parse_args()

    LEARNING_INTERACTIONS = 5e6
    SAMPLING_FREQUENCY = 4
    CONTROL_FREQUENCY = 4
    POPULATION_SIZE = NUM_PARALLEL_AGENT
    SIMULATION_TIME = int(LEARNING_INTERACTIONS / (CONTROL_FREQUENCY * POPULATION_SIZE))

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting learning")

    # random number generator
    rng = Random()
    rng.seed(42)

    # database
    database = open_async_database_sqlite("./RL_direct_tree/RLdatabases/test_1")

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    optimizer = RLOptimizer(
        rng=rng,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        simulation_time=SIMULATION_TIME,
        visualize=args.visualize
    )

    # initialize agent population
    agents = [make_agent() for _ in range(POPULATION_SIZE)]
    
    logging.info("Starting learning process..")

    await optimizer.train(agents, from_checkpoint=args.from_checkpoint)

    logging.info(f"Finished learning.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
