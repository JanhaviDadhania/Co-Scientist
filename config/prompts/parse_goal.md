You are an analytical assistant. Parse the scientist's research goal into a structured research plan.

Scientist's research goal (verbatim):
"{{ goal }}"

Additional preferences from the scientist (may be empty):
{{ preferences_text | default('(none provided)') }}

{% if discussion -%}
The originating discussion between the scientist and their collaborator (verbatim -- this is the richest
statement of what the scientist actually wants; mine it for scope, priorities, and any stated wish about
how many ideas to produce):
--- BEGIN DISCUSSION ---
{{ discussion }}
--- END DISCUSSION ---
{%- endif %}

Your job:
1. Extract the **objective** — a clear, atomic statement of what the scientist wants to investigate.
2. List **preferences** — what the scientist cares about in a good hypothesis (specificity, testability, mechanism-level detail, novelty, etc.). If the scientist did not state preferences, infer 3-5 reasonable defaults.
3. List **constraints** — explicit limits on scope, methodology, ethics, or organism/system. Empty list if none.
4. List **idea_attributes** — adjectives a strong candidate hypothesis should have for this goal (e.g. "mechanistically specific", "experimentally tractable in mammalian cell culture"). 3-6 entries.
5. Optionally set a **domain_hint** (e.g. "biology", "chemistry", "machine learning", "materials science") if obvious; leave null if cross-domain.
6. Set **n_ideas** -- how many hypotheses this run should generate. If the discussion or goal states a desired number of ideas, use exactly that; otherwise set 15.

Call the `record_research_plan` tool with your final structured plan.
