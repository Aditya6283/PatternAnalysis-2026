"""
safety.py - Automatic clinical-safety and readability checks for lay summaries.

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

Why: ROUGE only counts shared words. A summary can score well and still drop a
pneumothorax, invent a mass, turn "no effusion" into "fluid around the lungs", or turn
"suspected pneumonia" into "pneumonia" (the task's dilemma). This module turns those
error types into measurable flags.

How (a NegEx-style rule system [1], kept deliberately simple and transparent):
  1. A small lexicon maps the critical chest X-ray findings to BOTH expert terms
     ("pleural effusion") and the lay phrasings the human lay writers actually use
     ("fluid around the lungs"; collected from the training references). Ambiguous lay
     phrases (e.g. "collapsed lung", "shadow") are ignored in sentences that point to a
     different meaning ("...collapsed lung, called atelectasis", "the heart's shadow").
  2. Every mention is classified per sentence as affirmed or negated ("no", "there are no",
     "we didn't see", "... is not seen", "... has resolved"; but "no change in the effusion"
     is NOT a negation), and as hedged or definite ("possible", "suggests", "may", "could",
     "cannot be excluded").
  3. compare(source_report, summary) reports, for the critical findings:
        omission           affirmed in the report, not mentioned at all in the summary
        hallucination      affirmed in the summary, not mentioned at all in the report
        polarity flip      affirmed in one text, negated in the other
        certainty change   only hedged in one text, only definite in the other
                           ('suspected pneumonia' -> 'pneumonia' is an upgrade)
        laterality swap    the report puts a finding on one side, the summary on the other
        false reassurance  the summary says the whole study is normal although the
                           report affirms a critical finding
  Rule systems make mistakes, so evaluate.py first measures how often the checker fires
  on the HUMAN-written references (its false-alarm floor) before using it on model output.

Readability: Flesch-Kincaid grade level [2] (patient material targets about grade 6-8) and
a data-driven jargon rate (share of words that are typical of the expert reports but
avoided by the human lay writers).

References:
    [1] W. W. Chapman et al. A Simple Algorithm for Identifying Negated Findings and Diseases
        in Discharge Summaries. J. Biomedical Informatics 34(5), 2001 (NegEx).
    [2] J. P. Kincaid et al. Derivation of New Readability Formulas for Navy Enlisted
        Personnel. Research Branch Report 8-75, 1975 (Flesch-Kincaid grade level).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

# ---------------------------------------------------------------------------
# Lexicon: expert terms and lay phrasings per finding
# ---------------------------------------------------------------------------
# patterns  : always count as a mention of the concept
# ambiguous : count only if `veto` does not match the sentence
def _slot(n: int) -> str:
    """Up to n filler words, never a negation word (so 'is not enlarged' is not swallowed)."""
    return r"(?:(?!(?:not|no|nor|never|without)\b)[\w'-]+ ){0,%d}" % n


_ADJ = _slot(3)
CONCEPTS: Dict[str, Dict] = {
    "pneumothorax": {
        "critical": True,
        "patterns": [r"pneumothora\w*", r"air leak\w*",
                     r"air " + _slot(2) + r"(?:outside|around|surrounding) (?:of )?(?:the |a |both )?lungs?",
                     r"air in the (?:pleural|chest cavity|space around)"],
        "ambiguous": [r"collapsed lungs?"],
        "veto": r"atelecta|not fully|partial|\bpart\b|portion|piece|area of|areas of|thin|layer|segment|"
                r"lobe|band|plate|lamellar|laminar|bases?\b|bottom|lower part"},
    "pleural_effusion": {
        "critical": True,
        "patterns": [r"(?:pleural )?effusions?", r"hydrothorax", r"hemothorax",
                     r"(?:fluid|liquid) " + _ADJ + r"(?:around|surrounding|outside|between|next to) (?:of )?"
                     r"(?:the |both |each |his |her |a |their )?(?:[\w-]+ )?lungs?",
                     r"(?:fluid|liquid) " + _ADJ + r"(?:in|on|at|inside) (?:the |both )?" + _slot(3) +
                     r"(?:chest cavity|pleural space|lining of the lungs?|space around the lungs?)",
                     r"(?:fluid|liquid) " + _ADJ + r"(?:on|at|in) (?:both sides|each side|(?:the )?(?:left|right) side)"
                     r" of the (?:lungs?|chest)",
                     r"(?:trapped |loculated )?fluid collections?"]},
    "lung_opacity_infection": {
        "critical": True,
        "patterns": [r"consolidat\w*", r"infiltrat\w*", r"(?:broncho)?pneumoni\w*", r"opacit(?:y|ies)",
                     r"ground[- ]glass", r"air ?space (?:disease|opacit\w*)",
                     r"alveolar (?:pattern|filling|involvement)",
                     r"interstitial(?:-alveolar)? (?:pattern|involvement|markings|thickening|changes)",
                     r"(?:increased|increase in|more|higher|greater) " + _slot(2) + r"densit(?:y|ies)",
                     r"(?:lung|pulmonary) densit(?:y|ies)", r"denser",
                     r"(?:lung|chest|pulmonary) (?:super)?infections?", r"(?:super)?infection", r"infectious",
                     r"infections? (?:in|of|affecting) (?:the |both |his |her )?(?:[\w-]+ )?lungs?",
                     r"solidifi\w*", r"solid(?:ified)? (?:areas?|parts?|patch\w*|spots?|regions?)",
                     r"become (?:more )?solid", r"inflammat\w*",
                     r"(?:cloudy|cloudiness|hazy|haziness|blurry|blurred|unclear|fuzzy|foggy|whitish)"
                     r"(?: " + _slot(2) + r"(?:areas?|spots?|patch\w*|regions?|parts?|zones?))?",
                     r"fluid or (?:cells|inflammation|infection)", r"patch(?:es)? of fluid"],
        "ambiguous": [r"shadows?", r"white (?:areas?|spots?|patch\w*)"],
        "veto": r"heart|cardiac|nipple|breast|mammary|rib|bone|spine|vertebra|overlap|layer|structure|"
                r"soft tissue|mediastin|aorta|vessel|hila|hilum|diaphragm|clavicle|device|tube|"
                r"shadow of"},
    "cardiomegaly": {
        "critical": True,
        "patterns": [r"cardiomegal\w*", r"enlarged (?:cardiac|heart)\w*", r"cardiac enlargement",
                     r"enlargement of the (?:cardiac|heart)\w*",
                     r"(?:increased|increase in|enlarged) (?:the )?(?:size of the )?(?:cardiac|heart)\w* "
                     r"(?:silhouette|size|shadow|outline)",
                     r"(?:cardiac|heart)(?:'s)? (?:silhouette |size |shadow |outline )?(?:is |appears |looks |seems )?"
                     r"(?:slightly |mildly |moderately |markedly |significantly |very |a bit |somewhat |also )?"
                     r"(?:enlarged|increased in size)",
                     r"(?:increased|elevated|high) cardiothoracic (?:index|ratio)",
                     r"(?:heart|cardiac)(?: and chest| to chest| chest)? (?:size )?ratio " + _slot(2) +
                     r"(?:increased|high|elevated|large)",
                     r"heart\w*(?:'s)? " + _slot(5) + r"(?:larger|bigger|enlarged)",
                     r"heart\w*(?:'s)? (?:size|silhouette|shadow|outline) " + _slot(5) + r"(?:increased|larger|bigger)",
                     r"cardiothoracic (?:index|ratio) " + _slot(6) + r"(?:increased|elevated|high)",
                     r"(?:large|big|enlarged) heart",
                     r"(?:ratio|size) of (?:the )?heart\w* " + _slot(6) + r"(?:increased|larger|bigger|grown)"]},
    "nodule_mass": {
        "critical": True,
        "patterns": [r"(?:micro)?nodul\w*", r"(?<!solid )(?<!one )masse?s?", r"metasta\w*",
                     r"granuloma\w*", r"(?:round|rounded) (?:spots?|shadows?|lesions?|areas?)",
                     r"cancer (?:has )?spread\w*", r"spread(?:ing)? of (?:the )?cancer",
                     r"tumou?rs? in the lungs?"],
        "ambiguous": [r"(?:small|tiny|dense|calcified|high-density|little) (?:[\w-]+ )?spots?", r"bumps?",
                      r"lumps?"],
        "veto": r"cloudy|hazy|unclear|blurry|white|bone|spine|vertebra|rib\b|osteophyte|joint|skin|"
                r"lymph|thyroid|breast|axilla|armpit|liver|kidney|abdom|soft tissue|neck"},
    "fracture": {
        "critical": True,
        "patterns": [r"fractur\w*", r"broken " + _slot(2) + r"(?:bones?|ribs?|vertebra\w*|clavicle|collarbone)",
                     r"broken (?:part|piece|area|section)", r"(?:old |healed |healing |a |an )breaks?\b",
                     r"(?:rib|bone) injur\w*", r"(?:rib|bone) breaks?",
                     r"cracks? in (?:the |a )?" + _slot(2) + r"(?:bones?|ribs?)"]},
    "pulmonary_edema": {
        "critical": True,
        "patterns": [r"(?:pulmonary|interstitial|alveolar|lung|acute) o?edema", r"o?edema in the lungs?",
                     r"(?:heart|cardiac) (?:failure|insufficiency|decompensation)", r"\bchf\b",
                     r"(?:vascular|venous|pulmonary) congestion", r"congestive",
                     r"(?:fluid|water) (?:build ?-?up|overload|accumulation|leak\w*) " + _slot(3) +
                     r"(?:in|into) (?:the |both )?" + _slot(2) + r"lungs?",
                     r"(?:fluid|water) in (?:the |both |his |her )?lungs\b",
                     r"heart (?:is )?(?:not|isn'?t) (?:working|pumping)"]},
}

CRITICAL = tuple(k for k, v in CONCEPTS.items() if v["critical"])

# Whole-study "normal" statements (global subject only; organ-level "the heart looks normal"
# is not a global claim). Used for the false-reassurance flag.
GLOBAL_NORMAL = re.compile(
    r"^(?:conclusion:? |impression:? |findings:? )?(?:within normal limits|normal (?:study|chest x-?ray|"
    r"x-?ray|radiograph|exam\w*))\b|"
    r"\b(?<!other )no (?:significant |relevant |notable |remarkable |major |important |serious |"
    r"obvious |acute )*(?:radiolog\w* |patholog\w* |radiographic |medical )?(?:findings?|abnormalit\w*|"
    r"alterations?|problems?|issues?|concerns?)\b(?! (?:in|of|at|on|related|affecting))|"
    r"\b(?:everything|it all|all (?:of )?(?:the )?(?:results?|findings?)|the (?:chest x-?ray|x-?ray|"
    r"study|exam\w*|scan|images?|results?|report|radiograph|test|ct scan)) (?:\w+ )?(?:looks?|appears?|"
    r"seems?|is|was|are|were) (?:completely |totally |entirely |perfectly |generally )?normal\b|"
    r"\bnothing (?:unusual|abnormal|wrong|concerning|of concern)\b")

# ---------------------------------------------------------------------------
# Negation and hedging cues
# ---------------------------------------------------------------------------
_PRE_NEG = re.compile(
    r"\b(?:no|not|without|absence of|absent|negative for|free of|nor|neither|"
    r"there (?:is|are|was|were) no|no (?:signs?|evidence|indications?|traces?) of|"
    r"rul(?:e|ed|ing) out|to exclude|exclud(?:e|ed|ing)|r/o|check(?:ing)? (?:for|if)|"
    r"look(?:ing)? for|(?:did|does|do) not (?:show|see|find|reveal|appear)|"
    r"(?:didn|doesn|don|can|isn|aren|wasn|weren|haven|hasn)'?t|cannot see|"
    r"no longer|(?<!almost )(?<!nearly )(?<!partial )complete (?:resolution|disappearance) of|resolved)\b")
_PSEUDO_NEG = re.compile(  # "no change in the effusion" means the effusion is still there
    r"\bno (?:significant |relevant |notable |major |obvious |radiolog\w* |interval |other |big |real )*"
    r"(?:changes?|variations?|increase|decrease|worsening|improvement|progression|differences?|"
    r"modifications?)\b|\bnot (?:only|necessarily|changed|much)\b|\bwithout (?:significant |relevant )?"
    r"(?:changes?|variations?)\b|\bno (?:longer )?(?:has )?changed\b|"
    r"\b(?:hasn|haven|didn|doesn)'?t (?:changed?|improved?|worsened?|grown)\b")
_POST_NEG = re.compile(
    r"^\W*(?:\w+\W+){0,3}?(?:(?:is|are|was|were|has|have) (?:not|no longer)(?: been)? "
    r"(?:seen|observed|identified|visible|visuali[sz]ed|demonstrated|present|noted|detected|evident|shown|appreciated|found)|"
    r"(?:has|have) (?:now )?(?:resolved|disappeared|cleared|gone)|(?:is|are|was|were) (?:now )?(?:resolved|ruled out|excluded|gone|clear)|"
    r"clear(?:ing|ed|s)? up|(?:went|gone|going) away|(?:is|are) no longer (?:seen|there|present|visible)|"
    r"(?:isn|aren|wasn|weren)'?t (?:seen|there|present|visible)|not (?:seen|observed|identified|visible|visuali[sz]ed|demonstrated|present|detected))\b")
_HEDGE = re.compile(
    r"\b(?:possibl[ey]|probabl[ey]|likely|unlikely|suspect\w*|suggest\w*|may|might|could|"
    r"questionabl[ey]|doubtful|uncertain|indeterminate|equivocal|presum\w*|maybe|perhaps|"
    r"compatible with|cannot be (?:ruled out|excluded)|can'?t be (?:ruled out|excluded)|"
    r"not (?:be )?(?:ruled out|excluded)|differential|versus|vs)\b")
# 'cannot exclude pneumonia', "can't rule out a pneumothorax": the finding is possible, not absent
_HEDGED_EXCLUSION = re.compile(
    r"\b(?:cannot|can ?not|can'?t|unable to|could not|couldn'?t|(?:does|do|did) not|(?:doesn|don|didn)'?t)"
    r" (?:\w+ )?(?:exclude|rule out|discard)\b")
_POST_HEDGE = re.compile(  # 'an effusion is likely', 'pneumonia cannot be excluded'
    r"^\W*(?:\w+\W+){0,3}?(?:cannot be (?:ruled out|excluded)|can'?t be (?:ruled out|excluded)|"
    r"(?:is|are|was|were) (?:suspected|possible|likely|probable|questionable|not excluded|uncertain|"
    r"doubtful|not ruled out)|not (?:ruled out|excluded))\b")
# Words that end the scope of a negation / hedge cue inside a sentence
_TERMINATOR = re.compile(r"(?:;|:|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bexcept\b|\bwhereas\b|"
                         r"\bwhile\b|, with\b|\band (?:a|an|the|there|is|are|has|have|shows?)\b|"
                         r"\bwhich\b|\bwith (?:a|an)\b|\bbecause\b|\bdue to\b|\bcaused by\b|"
                         r"\brelated to\b|\bresulting in\b)")
# A new, positive alternative after a negation ends the negation (but still counts as a hedge):
# 'ruling out a tumour and more likely suggesting pneumonia' -> pneumonia is present (hedged).
_NEW_ALTERNATIVE = re.compile(r"\b(?:more|most) (?:likely|probabl[ey])\b|\brather\b|\binstead\b|"
                              r"\bfavou?ring\b|\bin favou?r of\b")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def split_sentences(text: str) -> List[str]:
    """Sentence splitter for clinical text (no external models needed). Expects normal
    capitalisation: a sentence ends at . ! ? followed by a capital letter or digit."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text).strip()) if s.strip()]


def _compile(patterns: Sequence[str]):
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b") if patterns else None


_COMPILED = {name: (_compile(c["patterns"]), _compile(c.get("ambiguous", [])),
                    re.compile(c["veto"]) if c.get("veto") else None)
             for name, c in CONCEPTS.items()}


@dataclass
class Mention:
    concept: str
    text: str
    negated: bool
    hedged: bool
    sides: frozenset = frozenset()   # {'left'}, {'right'}, {'both'} ... words next to the mention


@dataclass
class ConceptStatus:
    affirmed: bool = False   # at least one non-negated mention
    negated: bool = False    # mentioned, but only in negated form
    certainty: str = ""      # for affirmed concepts: 'hedged', 'definite' or 'mixed'
    sides: Set[str] = field(default_factory=set)   # laterality of the affirmed mentions
    mentions: List[Mention] = field(default_factory=list)


def _scope_before(sentence: str, start: int, max_words: int = 12) -> str:
    prefix = sentence[:start]
    cuts = [m.end() for m in _TERMINATOR.finditer(prefix)]
    if cuts:
        prefix = prefix[cuts[-1]:]
    return " ".join(prefix.split()[-max_words:])


def _scope_after(sentence: str, end: int, max_words: int = 8) -> str:
    suffix = sentence[end:]
    m = _TERMINATOR.search(suffix)
    if m:
        suffix = suffix[:m.start()]
    return " ".join(suffix.split()[:max_words])


def _negated_before(scope: str) -> bool:
    """True if the cue closest to the finding is a real negation: not 'no change in ...', and
    not followed by a new alternative ('ruling out a tumour, more likely pneumonia')."""
    cues = list(_PRE_NEG.finditer(scope))
    if not cues:
        return False
    last = cues[-1]
    for p in list(_PSEUDO_NEG.finditer(scope)) + list(_HEDGED_EXCLUSION.finditer(scope)):
        if p.start() <= last.start() < p.end():
            return False
    return not any(a.start() >= last.end() for a in _NEW_ALTERNATIVE.finditer(scope))


_SIDE_WORDS = {"left": "left", "right": "right", "both": "both", "bilateral": "both", "bilaterally": "both",
               "bibasal": "both", "bibasilar": "both", "biapical": "both"}


def _sides(before: str, matched: str, after: str) -> frozenset:
    """Laterality words close to a mention: 4 words before, inside it, 3 words after."""
    near = before.split()[-4:] + matched.split() + after.split()[:3]
    return frozenset(_SIDE_WORDS[w.strip(".,;:()")] for w in near if w.strip(".,;:()") in _SIDE_WORDS)


def find_mentions(text: str) -> List[Mention]:
    """All critical-finding mentions in `text`, each marked negated / hedged."""
    out = []
    for raw_sentence in split_sentences(text):
        sentence = raw_sentence.lower()
        for concept, (pattern, ambiguous, veto) in _COMPILED.items():
            matches = list(pattern.finditer(sentence)) if pattern else []
            if ambiguous is not None and not (veto and veto.search(sentence)):
                matches += list(ambiguous.finditer(sentence))
            for m in matches:
                before = _scope_before(sentence, m.start())
                after = _scope_after(sentence, m.end())
                negated = _negated_before(before) or bool(_POST_NEG.search(after))
                hedged = (bool(_HEDGE.search(before)) or bool(_HEDGED_EXCLUSION.search(before))
                          or bool(_POST_HEDGE.search(after)))
                out.append(Mention(concept, m.group(0), negated, hedged, _sides(before, m.group(0), after)))
    return out


def concept_status(text: str) -> Dict[str, ConceptStatus]:
    """Per-concept summary of a text's mentions (only concepts that are mentioned)."""
    status: Dict[str, ConceptStatus] = {}
    for mention in find_mentions(text):
        status.setdefault(mention.concept, ConceptStatus()).mentions.append(mention)
    for st in status.values():
        affirmed = [m for m in st.mentions if not m.negated]
        st.affirmed = bool(affirmed)
        st.negated = not affirmed
        if affirmed:
            hedges = {m.hedged for m in affirmed}
            st.certainty = "mixed" if len(hedges) == 2 else ("hedged" if True in hedges else "definite")
            st.sides = set().union(*(m.sides for m in affirmed))
    return status


_ORGAN = re.compile(r"\b(?:lungs?|heart|cardiac|bones?|ribs?|spine|vertebra\w*|mediastin\w*|parenchyma|"
                    r"pleura\w*|aorta|hil(?:a|um|ar)|abdom\w*|thyroid|liver|kidney\w*|breast|soft tissue)")


def is_global_normal(text: str) -> bool:
    """Does the text say the whole study is normal ('Everything looks normal.')?
    Organ-level statements ('the lungs look fine with no significant issues') do not count."""
    return any(GLOBAL_NORMAL.search(s.lower()) and not _ORGAN.search(s.lower())
               for s in split_sentences(text))


def compare(source: str, summary: str, concepts: Sequence[str] = CRITICAL) -> Dict:
    """Clinical-safety flags of `summary` against the `source` report (see module docstring).
    Each list holds the concepts concerned; 'any' is True if at least one flag fired."""
    s_src, s_sum = concept_status(source), concept_status(summary)
    flags = {"omission": [], "hallucination": [], "polarity_flip": [],
             "certainty_upgrade": [], "certainty_downgrade": [], "laterality_swap": []}
    for c in concepts:
        a, b = s_src.get(c), s_sum.get(c)
        if a and a.affirmed and b is None:
            flags["omission"].append(c)
        elif b and b.affirmed and a is None:
            flags["hallucination"].append(c)
        elif a and b and a.affirmed != b.affirmed:
            flags["polarity_flip"].append(c)
        elif a and b and a.affirmed and b.affirmed:
            if a.certainty == "hedged" and b.certainty == "definite":
                flags["certainty_upgrade"].append(c)
            elif a.certainty == "definite" and b.certainty == "hedged":
                flags["certainty_downgrade"].append(c)
            # wrong side: the report names exactly one side and the summary exactly the other
            if a.sides in ({"left"}, {"right"}) and b.sides in ({"left"}, {"right"}) and a.sides != b.sides:
                flags["laterality_swap"].append(c)
    src_positive = any(s_src[c].affirmed for c in concepts if c in s_src)
    sum_positive = any(s_sum[c].affirmed for c in concepts if c in s_sum)
    flags["false_reassurance"] = bool(src_positive and not sum_positive and is_global_normal(summary))
    flags["any"] = bool(any(flags[k] for k in FLAG_TYPES[:-1]) or flags["false_reassurance"])
    return flags


FLAG_TYPES = ("omission", "hallucination", "polarity_flip", "certainty_upgrade",
              "certainty_downgrade", "laterality_swap", "false_reassurance")  # false_reassurance stays last


def finding_set(text: str, concepts: Sequence[str] = CRITICAL) -> Dict[str, str]:
    """{concept: 'pos' | 'neg'} for the critical concepts a text mentions."""
    st = concept_status(text)
    return {c: ("pos" if st[c].affirmed else "neg") for c in concepts if c in st}


def disagrees(text_a: str, text_b: str, concepts: Sequence[str] = CRITICAL) -> bool:
    """True if two texts disagree on whether any critical finding is present.

    Absent and negated count as the same ('not present'): 'no effusion' vs. silence is
    fine, but 'effusion' vs. silence or vs. 'no effusion' is a disagreement. Used to label
    model summaries against the human reference (an offline audit label).
    """
    a, b = finding_set(text_a, concepts), finding_set(text_b, concepts)
    return any((a.get(c) == "pos") != (b.get(c) == "pos") for c in concepts)


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Heuristic English syllable count (vowel groups, silent final -e / -es / -ed)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", lambda m: m.group(0)[0], w)
    w = re.sub(r"^y", "", w)
    return max(1, len(_VOWEL_GROUPS.findall(w)))


def words(text: str) -> List[str]:
    return _WORD_RE.findall(str(text))


def fkgl(text: str) -> float:
    """Flesch-Kincaid grade level: 0.39 (words/sentence) + 11.8 (syllables/word) - 15.59."""
    w = words(text)
    if not w:
        return float("nan")
    n_sent = max(1, len(split_sentences(text)))
    syll = sum(count_syllables(x) for x in w)
    return 0.39 * len(w) / n_sent + 11.8 * syll / len(w) - 15.59


def build_jargon_vocabulary(expert_texts: Iterable[str], lay_texts: Iterable[str],
                            top_k: int = 300, min_count: int = 20) -> Set[str]:
    """Words much more frequent in expert reports than in the human lay rewrites.

    Ranked by the smoothed log-odds ratio log((e + 1) / (E + V)) - log((l + 1) / (L + V)),
    keeping the `top_k` most expert-specific words (length >= 4) seen at least `min_count`
    times. Computed on the TRAINING split only.
    """
    e_counts = Counter(w.lower() for t in expert_texts for w in words(t))
    l_counts = Counter(w.lower() for t in lay_texts for w in words(t))
    vocab = set(e_counts) | set(l_counts)
    E, L, V = sum(e_counts.values()), sum(l_counts.values()), len(vocab)
    scores = {}
    for w in vocab:
        if e_counts[w] + l_counts[w] < min_count or len(w) < 4:
            continue
        scores[w] = math.log((e_counts[w] + 1) / (E + V)) - math.log((l_counts[w] + 1) / (L + V))
    return {w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]}


def jargon_rate(text: str, jargon: Set[str]) -> float:
    w = [x.lower() for x in words(text)]
    return sum(x in jargon for x in w) / len(w) if w else float("nan")
