from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='Airline_sentiment',
      version="1.0",
      description="Airline tweet sentiment model",
      author="Stuart Howarth",
      author_email="stuarthowarth88@gmail.com",
      install_requires=requirements,
      packages=find_packages())
