=============================
Architecture and verification
=============================

Rizemind addresses two related tasks: training across separate data
environments, and maintaining attributable records around that collaboration.

The collaboration problem
=========================

Consider three organizations with data relevant to the same modelling task. A
federated workflow allows each organization to train locally and exchange
permitted model updates rather than first combine its raw dataset with the
others.

Independent participants also need to understand who submitted an update, which
round it belongs to, how contributions are measured, and which records can be
checked. Rizemind adds mechanisms around the federated workflow for these tasks.

Three functions
===============

**Data and compute:** participants operate local training against their own
source datasets.

**Federated workflow:** Flower and the configured Rizemind components coordinate
updates, validation, aggregation and evaluation.

**Verification:** configured components authenticate updates and record selected
training metadata. A ledger record is distinct from the underlying dataset and
from the model artifact itself.

What Rizemind adds
==================

**Authentication and signed updates.** Cryptographic signatures associate an
update with a signing identity and allow the signed material to be checked. See
:doc:`signature authentication <../how-tos/web3-auth/index>` and
:doc:`the model-signing reference <../references/authentication/notary/model/index>`.
A signature establishes a relationship to a key; organizational identity and
permission to participate require the deployment's own identity and access
policy.

**Contribution measurement.** Configured strategies evaluate participant
contributions against the selected task, metric and evaluation procedure. A
contribution score is a result under that method, not an unconditional measure
of a participant's value. See :doc:`Strategies <../references/strategies/index>`.
Contribution measurement and any compensation mechanism are separate concerns.

**Training provenance.** Recorded identifiers and metadata can relate
participants, rounds, updates, model artifacts and contribution results. See
:ref:`model identity and records <rizemind-model-records>` for how to
distinguish the resulting artifact from a record referring to it.

**Ledger-backed records.** Selected events or commitments can be recorded
onchain. The :doc:`Arc guide <../quickstarts/arc-mainnet>` identifies the
deployed environment and the verification steps for that example.

.. _rizemind-model-records:

Model identity, provenance and usage
====================================

Model identity specifies which model and version a record refers to. Training
provenance describes how a version was produced. Usage concerns which version
was used in a recorded activity. These are different questions.

When inspecting a record, distinguish a model used as input to an activity from
a model produced as output. Interpret its identifiers and checks according to
the fields actually captured by the configured workflow. A model reference alone
does not establish a subsequent application use.

.. _rizemind-data-boundaries:

Data and visibility boundaries
==============================

**Local:** raw training datasets, local data access, compute and credentials
remain under the participant's administration.

**Exchanged:** permitted updates, parameters, evaluation outputs and other
protocol information follow the configured federated workflow.

**Recorded:** identifiers, events, signatures or commitments are exposed only as
specified by the selected implementation. Review the exact fields before writing
them to a public ledger.

**Stored elsewhere:** metrics and model artifacts may be sent to a configured
tracking or artifact service. For example, the
:doc:`MLflow storage module <../references/logging/mlflow/metric_storage>`
supports logging models and metrics to a tracking server. Local raw data does
not mean all derived artifacts remain local.

Raw training datasets are not required onchain. The implementation and its
configuration determine the visibility of updates, evaluation outputs and
metadata.

Privacy and verification limits
===============================

Keeping source datasets local reduces the need to transfer them; it does not by
itself guarantee that updates or derived models disclose no information.
Deployment controls may include access restrictions, transport protection,
secure aggregation or differential privacy, where supported and configured. Do
not assume a protection is enabled because it is available in the broader
Flower ecosystem.

For background, see Flower's
`differential privacy explanation <https://flower.ai/docs/framework/explanation-differential-privacy.html>`_
and
`secure aggregation explanation <https://flower.ai/docs/framework/explanation-ref-secure-aggregation-protocols.html>`_.

A signed update or recorded commitment does not, by itself, prove that a model
is accurate, that an inference executed correctly, or that a particular model
was used in a production system. Those statements require their own mechanisms
and evidence.

In these docs, **sovereign data** describes retaining control over source
datasets. It is not a blanket claim of regulatory compliance, data residency or
complete infrastructure independence. Each organization must assess its
deployment's requirements and controls.

Scope
=====

Rizemind is a federated-learning framework with specialized coordination,
authentication, contribution-measurement and verification components. It is not
a centralized data marketplace, a proprietary foundation model, or an agent
runtime. Blockchain records do not make models or data private by themselves.

Next: :doc:`Illustrative use cases <use-cases>`,
:doc:`Integrating with Flower <../how-tos/flower-integration>`, or
:doc:`Rizemind on Arc Mainnet <../quickstarts/arc-mainnet>`.
