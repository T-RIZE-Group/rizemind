=====================================================================================================
A short reference for ReStructuredText (Document Title - no new line allowed - no text prior allowed)
=====================================================================================================
-----------------------------------------------------------------
Primarily used for a quick start (Subtitle - no new line allowed)
-----------------------------------------------------------------



Paragraphs
==========
Paragraphs are split with a newline. For example this is paragraph 1.

Now this is Paragraph 2!


Text Styles
===========
*italic*
**bold**

To use * itself, just use \*


Lists
=====
There are three types of lists in ReST:

Enumerated Lists
----------------
Simply use 1, I, i, A. ,a. ,(1), or 1). Example:
1. Wow!
2. Brilliant
    I. Now a sublist!
    II. Has two lines!!!
        i. Another subsublist???!!!
        ii. coool

Bulleted Lists
--------------
Simply use -, +, or *
They can also be combined together, for example:
* A bullet list
    - A sublist
        + A subsublist
    - Another sublist


Definition Lists
----------------
Definitions
  Definitions consist of a term, and its explanation, like the one that you see right now!


Code Samples
============
To use a preformatted text in line use ``code`` or for a block simply put ``::`` and then make an indentation and put the code block. Example::
    This is coooode!
    *bold?* Nope!
And now *everything* is back to normal.



Sections
========
Every document must have a title. The following roles have caveats, but for the sake of consistency, let's follow the following principles::
    ==============
    Document Title
    ==============
    --------
    Subtitle
    --------

    Section 1
    =========

    SubSection 1.1
    -----------

    SubSubSection 1.1.1
    ~~~~~~~~~~~~~~~~~~~

If necessary, for more subsubsubsections and deeper indentations, any of the following characters can be used:
``: ' " ^ _ * + # < >``


References
==========
For further reading please visit the Quickstart_ or Quickref_.

.. _Quickstart: https://docutils.sourceforge.io/docs/user/rst/quickstart.html
.. _Quickref: https://docutils.sourceforge.io/docs/user/rst/quickref.html

