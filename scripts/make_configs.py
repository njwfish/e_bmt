import click


@click.group()
def main():
    pass


@main.command()
@click.option('--data_path', multiple=True, type=click.Path())
@click.option('--agent_path', multiple=True, type=click.Path())
@click.option('--out_dir')
def generate(data_paths, agent_paths, out_dir):
    data_configs = [
        config for path in data_paths for config in json.load(open(path))
    ]
    agent_configs = [
        config for path in agent_paths for config in json.load(open(path))
    ]
