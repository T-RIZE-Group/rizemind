=======================
Integrating with Flower
=======================

Rizemind complements a Flower-based federated workflow. You bring the model,
training procedure and evaluation metrics, then configure the relevant Rizemind
components around participant authentication, contribution measurement and
records.

This page explains the integration sequence. Follow the version-specific
:doc:`quickstarts <../quickstarts/index>` for executable instructions.

.. _rizemind-federation-roles:

Roles and responsibilities
==========================

**Model developer:** defines the model, data interface, training procedure,
evaluation metrics and suitable aggregation approach.

**Participating organization:** operates its local participant, approves access
to local data, manages credentials and reviews what the workflow may disclose.

**Federation coordinator:** configures the task, participant policy,
aggregation, evaluation and record-keeping environment. Specify who controls the
aggregator and who is permitted to submit transactions for the chosen
deployment.

**Ledger infrastructure:** hosts the configured contracts and records. This role
is distinct from operating a model-training client or a federation coordinator.

Integration sequence
====================

1. Start with the existing model and define its training and evaluation
   procedure.
2. Define the federation: participants, task, metrics and the permitted data
   flows.
3. Deploy local participant components using the selected example's
   configuration.
4. Train against each participant's local dataset.
5. Authenticate submitted updates through the configured signing mechanism.
6. Validate and aggregate permitted updates using the selected strategy.
7. Measure contributions where the chosen configuration enables that function.
8. Record the selected events and metadata in the configured verification
   environment.
9. Evaluate the resulting model version and decide whether to run the next
   round.

Distinguish the model used as input from the version produced by the workflow.
See :ref:`model identity and records <rizemind-model-records>` for that
distinction.

Relevant references:
:doc:`Authentication <../references/authentication/index>`,
:doc:`signed model updates <../references/authentication/notary/model/index>`,
:doc:`Strategies <../references/strategies/index>`,
:doc:`Configuration <../references/configuration/index>` and
:doc:`Web3 <../references/web3/index>`.

Flower ecosystem and compatibility
==================================

Flower provides examples for multiple modelling frameworks and workloads. Its
example catalogue is a useful starting point; it is not a list of workflows
already tested with every Rizemind module or on Arc.

Use the prerequisites of the selected Rizemind example. Check the Flower
version, client and strategy interfaces, serialization, signing and
contribution-evaluation requirements before adapting another workload.

* `Flower documentation <https://flower.ai/docs/>`_
* `Flower examples <https://flower.ai/docs/examples/>`_
* `Flower PyTorch tutorial <https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html>`_

Start building
==============

New to Rizemind? :doc:`Install the package <../install>` and choose
:doc:`an example <../quickstarts/index>`.

Review :doc:`signature authentication <web3-auth/index>` for that feature's
integration pattern. Keep its documented network and example assumptions intact.

For Arc-specific prerequisites and execution, follow
:doc:`Rizemind on Arc Mainnet <../quickstarts/arc-mainnet>`, including its
testnet rehearsal and funding warnings.
