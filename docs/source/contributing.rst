.. _contributing:

Contributing
================

The project is open to community contributions. Feel free to open an `issue <https://github.com/esa/torchquad/issues>`_ 
or write us an `email <https://torchquad.readthedocs.io/en/main/contact.html#feedback>`_ if you would like to discuss a problem or idea first.

If you want to contribute, please 

1. Fork the project on `GitHub <https://github.com/esa/torchquad>`_. 
2. Get the most up-to-date code by following this quick guide for installing *torchquad* from source:

a. Get `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or similar
b. Clone the repo

   .. code-block:: bash

      git clone https://github.com/esa/torchquad.git

c. With the default configuration, all numerical backends with CUDA support are installed. If this should not happen, comment out unwanted packages in ``environment.yml``.

d. Set up the environment. This creates a conda environment called ``torchquad`` and installs the required dependencies.

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate torchquad

Once the installation is done, then you are ready to contribute. 
Please note that PRs should be created from and into the ``develop`` branch. For each release the develop branch is merged into main.

3. Create your Feature Branch: ``git checkout -b feature/AmazingFeature``
4. Commit your Changes: ``git commit -m 'Add some AmazingFeature'``
5. Push to the Branch: ``git push origin feature/AmazingFeature``
6. Open a Pull Request on the ``develop`` branch, *not* ``main`` (NB: We autoformat every PR with black. Our GitHub actions may create additional commits on your PR for that reason.)

and we will have a look at your contribution as soon as we can. 

Furthermore, please make sure that your PR passes all automated tests. Review will only happen after that.
Only PRs created on the ``develop`` branch with all tests passing will be considered. The only exception to this rule is if you want to update the documentation in relation to the current release on ``conda`` / ``pip``. 
In that case you may ask to merge directly into ``main``.
