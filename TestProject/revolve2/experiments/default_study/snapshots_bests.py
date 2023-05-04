from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer
from revolve2.core.modular_robot.render.render import Render
#TODO: make import based on param and move file to anal_resources
from genotype import GenotypeSerializer, develop
from revolve2.core.database.serializers import DbFloat

import os


async def main() -> None:

    study = 'default_study'
    experiments_name = ['joints']
    runs = list(range(1, 10+1))
    generations = [37]

    for experiment_name in experiments_name:
        print(experiment_name)
        for run in runs:
            print(' run: ', run)

            path = f'/storage/karine/{study}/analysis/snapshots/{experiment_name}/run_{run}'
            if not os.path.exists(path):
                os.makedirs(path)

            db = open_async_database_sqlite(f'/storage/karine/{study}/{experiment_name}/run_{run}')

            for gen in generations:
                print('  gen: ', gen)
                path_gen = f'{path}/gen_{gen}'
                if os.path.exists(path_gen):
                    print(f'{path_gen} already exists!')
                else:
                    os.makedirs(path_gen)

                    async with AsyncSession(db) as session:

                        rows = (
                            (await session.execute(select(DbEAOptimizer))).all()
                        )
                        max_modules = rows[0].DbEAOptimizer.max_modules
                        substrate_radius = rows[0].DbEAOptimizer.substrate_radius

                        rows = (
                            (await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat)
                                                   .filter(DbEAOptimizerGeneration.generation_index.in_([gen]))
                                                   .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                                           & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                                           )
                                                   .order_by(DbFloat.speed_x.desc())


                            )).all()
                        )

                        for idx, r in enumerate(rows):
                            #print('geno',r.DbEAOptimizerIndividual.genotype_id)
                            genotype = (
                                await GenotypeSerializer.from_database(
                                    session, [r.DbEAOptimizerIndividual.genotype_id]
                                )
                            )[0]

                            phenotype = develop(genotype, genotype.mapping_seed, max_modules, substrate_radius)
                            render = Render()
                            img_path = f'{path_gen}/{idx}_{r.DbEAOptimizerIndividual.individual_id}.png'
                            render.render_robot(phenotype.body.core, img_path)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# can be run from root