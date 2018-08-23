import setuptools

setuptools.setup(
    name="lm_ltr",
    version="0.1.0",
    url="none",

    author="Dany Haddad",
    author_email="danyhaddad43@gmail.com",

    description="Learning to rank by fine tuning an LM",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
