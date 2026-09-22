.. rizemind documentation master file, created by
   sphinx-quickstart on Wed Feb 19 10:00:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Meet Rizemind
=================

.. toctree::
   :maxdepth: 2
   :caption: Sections
   :hidden:

   concepts/index
   install
   quickstarts/index
   how-tos/index
   references/index
   research/index
   developers/index

**Shared intelligence. Sovereign data.**

Rizemind is an open-source framework for **federated learning with verifiable
multi-party coordination**.

It is designed for developers and organizations that need to train or evaluate
models across separate data environments without transferring the underlying
datasets into a common repository.

Each participant trains locally. Raw training data remains in the participant's
environment. Authorized model updates are exchanged through the
federated-learning process.

Rizemind adds participant authentication, cryptographically signed model
updates, contribution measurement and ledger-backed records of selected training
events. The available records and controls depend on the configured workflow.
See :ref:`model identity and records <rizemind-model-records>` for how to
interpret the model and version associated with an activity.

**Arc deployment:** Rizemind's ledger-backed components are deployed on Arc
Mainnet. See :doc:`the deployment guide <quickstarts/arc-mainnet>` for the
implementation, network configuration and verification steps.

Start here
==========

* :doc:`Install Rizemind <install>`
* :doc:`Run an example <quickstarts/index>`
* :doc:`Understand the architecture <concepts/architecture>`
* `View the source code <https://github.com/T-RIZE-Group/rizemind>`_

.. _flower-trize:
.. _flower-t-rize:

Built on Flower
===============

Rizemind builds on `Flower <https://flower.ai/docs/>`_. Flower provides the
federated-learning foundation; Rizemind adds infrastructure for coordinating
independent participants. Developers bring their own model, training procedure
and evaluation metrics rather than adopt a proprietary Rizemind model.

See :doc:`Integrating with Flower <how-tos/flower-integration>` for the
adoption workflow and the distinction between Flower examples and tested
Rizemind support.

Choose your path
================

**Developers:** start with an existing model and review
:doc:`the integration workflow <how-tos/flower-integration>`.

**Participating organizations:** review
:ref:`the data and visibility boundaries <rizemind-data-boundaries>` before
connecting a local participant to a federation.

**Federation coordinators:** review
:ref:`the roles and responsibilities <rizemind-federation-roles>`, then select
an appropriate :doc:`example <quickstarts/index>`.

.. _why-cooperation:
.. _design-principles:
.. _how-it-works:
.. _key-capabilities:

Further explanation
===================

Read :doc:`the architecture <concepts/architecture>` for the workflow,
capabilities and information boundaries. Explore
:doc:`illustrative use cases <concepts/use-cases>` and
:doc:`the research foundation <research/index>`.

.. _get-involved:

Community and security
======================

* `Source code <https://github.com/T-RIZE-Group/rizemind>`_
* `Community Slack <https://join.slack.com/t/rizemind/shared_invite/zt-3dufpugzb-znhIxQcO8sCAKY6V6JrhCg>`_
* `Report a security concern <mailto:security@t-rize.io>`_
