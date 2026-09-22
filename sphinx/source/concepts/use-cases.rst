======================
Illustrative use cases
======================

These scenarios explain where the architecture may be useful. They are not
statements of customer deployments, measured model improvements, or features
enabled in every Rizemind configuration.

An AI vendor and financial institutions
=======================================

Consider a vendor supplying a transaction-monitoring model to several financial
institutions. Each institution observes different transactions and edge cases.

A federated implementation can keep training against those datasets within each
institution. Permitted updates are then aggregated. Rizemind can associate
submitted updates with signing identities, measure contributions under the
chosen strategy and record selected training events.

The vendor can coordinate collaborative model development without requiring a
central repository containing every institution's underlying training dataset.
The parties still need a compatible task, agreed evaluation metrics and suitable
security controls. Improvement is an evaluation result, not an automatic
property of federation.

Models used by agents
=====================

The same architecture may support models used inside agents. Different
deployments can produce local training or evaluation signals that organizations
do not want to export as source data.

Approved training can occur locally while the federation exchanges permitted
updates. Depending on the recorded fields, the workflow can associate updates
with participants, model versions, rounds and contribution results. See
:ref:`model identity and records <rizemind-model-records>` for how to interpret
those associations.

This is the federated training and verification layer around models that may be
used by agents. It is not the agent runtime, an agent-governance platform, or an
automatic record of every inference or downstream use.

Public-sector and multi-entity organizations
============================================

Several public healthcare institutions might contribute to the same analytical
modelling task while maintaining separate source datasets. Similar boundaries
can exist between business units, subsidiaries or geographic environments within
one organization.

A federated architecture can execute approved training locally and aggregate
permitted updates. Rizemind adds authentication, attribution and selected
training records around that process.

Suitability depends on the actual data, model, permissions and deployment
controls. Rizemind does not itself establish compliance with healthcare,
privacy, residency or other sector requirements.

Next steps
==========

Review :ref:`the data boundaries <rizemind-data-boundaries>` and
:doc:`the integration workflow <../how-tos/flower-integration>`. For a
documented technical example, use
:doc:`Rizemind on Arc Mainnet <../quickstarts/arc-mainnet>`. The Arc example is
separate from the illustrative institutional scenarios above.
