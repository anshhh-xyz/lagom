import json
from pathlib import Path
from collections import Counter

HELD_OUT = Path(r"d:/Projects/Lagom/eval/held_out.jsonl")

ESSAYS = [
    ("artificial_intelligence_future",
     "In the rapidly evolving landscape of artificial intelligence, the intersection of machine learning and human cognition stands as a paramount milestone in our technological journey. As society navigates this transformative paradigm shift, understanding the multifaceted relationship between intelligent systems and human decision-making becomes increasingly essential. AI is no longer a mere auxiliary instrument; rather, it has seamlessly integrated into the very fabric of modern enterprise, driving unprecedented efficiency and fostering innovation across geographical boundaries.\n\nAt the core of this monumental evolution lies the profound impact on data-driven decision-making. The rise of sophisticated neural architectures has effectively dismantled traditional barriers to knowledge discovery, empowering organizations to extract actionable insights from vast datasets instantaneously. This unprecedented capability nurtures a vibrant ecosystem of analytical intelligence, enabling cross-disciplinary synergy and accelerating intellectual progress in domains ranging from healthcare to financial systems.\n\nFurthermore, AI integration significantly transforms the workforce landscape. Adaptive automation frameworks, augmented by deep learning algorithms, democratize access to high-level cognitive assistance, catering to diverse professional needs and individual workflows. In organizational spheres, AI-driven pipelines optimize operational productivity, allowing individuals to transcend repetitive tasks and redirect focus toward creative problem-solving, strategic foresight, and ethical reflection.\n\nHowever, navigating this advanced landscape necessitates a balanced approach. While the benefits of rapid AI innovation are vast, they must be harmonized with robust ethical frameworks, rigorous data governance safeguards, and an unwavering commitment to algorithmic equity. Ensuring that progress remains inclusive and aligned with human-centric values is pivotal to building a sustainable future for all.\n\nIn conclusion, the symbiotic relationship between artificial intelligence and human society continues to redefine the boundaries of what is possible. By thoughtfully embracing innovation while maintaining a resolute focus on ethical stewardship, humanity can usher in a harmonious era of sustained growth, nuanced understanding, and holistic progress across all sectors of civilization.")
    ,("climate_change_policy",
     "In the rapidly evolving landscape of global climate governance, the intersection of environmental science and policy-making stands as a paramount challenge of our era. As society navigates this transformative paradigm shift, understanding the multifaceted relationship between industrial development and ecological sustainability becomes increasingly essential. Climate policy is no longer a peripheral concern; rather, it has become seamlessly integrated into the very tapestry of international diplomacy and economic planning, driving robust systemic reform.\n\nAt the core of this monumental challenge lies the profound impact of carbon emissions on global temperature trajectories. The rise of sophisticated climate modeling has effectively dismantled traditional barriers to predictive understanding, empowering policymakers to develop comprehensive mitigation strategies with unprecedented precision. This analytical capability nurtures a vibrant ecosystem of evidence-based governance, enabling cross-border synergy and accelerating the intellectual progress necessary for systemic change.\n\nFurthermore, climate policy integration significantly transforms the energy landscape. Personalized clean energy pathways, augmented by adaptive grid technologies, democratize access to renewable power, catering to diverse community needs and individual consumption patterns. In industrial spheres, automated efficiency protocols optimize resource utilization, allowing enterprises to transcend legacy carbon dependencies and redirect investment toward sustainable innovation.\n\nHowever, navigating this advanced landscape necessitates a balanced approach. While the benefits of rapid decarbonization are vast, they must be harmonized with careful consideration of economic equity, robust social safety nets, and an unwavering commitment to just transition for affected communities and workers across the global economy.\n\nIn conclusion, the symbiotic relationship between humanity and the natural world continues to redefine our understanding of progress. By thoughtfully embracing systemic change while maintaining a resolute focus on ethical stewardship, the global community can usher in a harmonious era of ecological restoration, economic resilience, and holistic well-being for all inhabitants of our shared planet.")
]

records = []
with open(HELD_OUT, encoding="utf-8") as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))

for topic, text in ESSAYS:
    text = text.strip()
    ai_w = len(text.split())
    tw = int(round(ai_w * 0.85, -1))
    records.append({"text": text, "category": "essay", "source": "synthetic_ai_essay",
        "topic": topic, "human_reference": None, "ai_words": ai_w, "target_words": tw})
    print("Added", topic, ":", ai_w, "words")

with open(HELD_OUT, "w", encoding="utf-8") as f:
    for rec in records: f.write(json.dumps(rec, ensure_ascii=False) + chr(10))

print("Total:", len(records))
