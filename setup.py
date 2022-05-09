from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


setup(
    name='hdpgmm',
    version='0.0.1',
    description='hdpgmm implementation based on pytorch',
    long_desription=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    url='https://github.com/eldrin/pytorch-hdpgmm',
    author='Jaehun Kim',
    author_email='j.h.kim@tudelft.nl',
    license='MIT',
    packages=find_packages('.'),
    install_requires=requirements(),
    test_suite='tests',
    zip_safe=False
)
