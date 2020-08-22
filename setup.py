from setuptools import setup, find_packages

setup(
    name='taglets',
    version='0.0.1',
    packages=find_packages(),
    dependency_links=[
        'git+https://github.com/BatsResearch/labelmodels.git'
    ],
    entry_points={"console_scripts": ["taglets = taglets.task.jpl:ext_launch"]},

)
