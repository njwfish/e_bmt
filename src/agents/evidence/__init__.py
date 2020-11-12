"""Evidence modules that keep track of evidence against each hypothesis that
has been collected so far.

These will generally be in the form of e or p-values.
"""

import agents.evidence.base
import agents.evidence.e.pmh
import agents.evidence.p.lcb
import agents.evidence.p.etop
