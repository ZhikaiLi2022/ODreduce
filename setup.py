import re
import setuptools

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project+'/__init__.py').read())
    return result.group(1)

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setuptools.setup(
    name="odreduce",
    version=get_property('__version__', 'odreduce'),
    license="MIT",
    author="group 1",
    author_email="lzk170@163.com",
    description="Observation and Data Reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZhikaiLi2022/odreduce",
    project_urls={
        "Source": "https://github.com/ZhikaiLi2022/odreduce",
        "Bug Tracker": "https://github.com/ZhikaiLi2022/odreduce/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    packages=setuptools.find_packages(),
    package_data={"": ["dicts/*.dict"]},
    entry_points={'console_scripts':['odreduce=odreduce.cli:main']},
    python_requires=">=3.6",
)
