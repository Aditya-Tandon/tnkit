API reference
=============

Generated from the installed package on every build. The :doc:`public` page lists every
name ``tnkit`` exports; :doc:`blocks` covers the block-sparse tensor modules and
:doc:`symmetric` the symmetric-tensor modules (SU(2) and other non-abelian symmetries), both
provisional; :doc:`opsum` covers the compiler from operator sums to MPOs, and :doc:`thermal`
thermal states by purification.

Settings can be passed to the solvers either as keyword arguments or grouped in an options
object (``DMRGOptions``, ``TDVPOptions``, ...). The two styles run the same code.

.. toctree::
   :maxdepth: 1

   public
   blocks
   symmetric
   opsum
   thermal
