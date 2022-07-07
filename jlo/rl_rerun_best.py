from GA.genotype import GenotypeSerializer, develop
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from RL.rl_rerunner import AgentRerunner

async def main() -> None:

    rerunner = AgentRerunner()
    await rerunner.rerun()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
