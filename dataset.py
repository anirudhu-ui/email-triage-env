"""
dataset.py
----------
Expanded email dataset for EmailTriageEnv.
20 emails across 3 difficulty tiers: easy, medium, hard (ambiguous).

Each email has:
    text     : str   — full email body with From/Subject
    label    : str   — spam | important | promotional
    priority : str   — low | medium | high
    reply    : str   — ignore | acknowledge | respond
    tier     : str   — easy | medium | hard
    reason   : str   — why this label is correct (for evaluation/debugging)
"""

EMAIL_DATASET = [

    # ══════════════════════════════════════════════════════
    # TIER: EASY — obvious signals, no ambiguity
    # ══════════════════════════════════════════════════════

    {
        "text": (
            "From: manager@work.com\n"
            "Subject: Project Deadline — URGENT\n\n"
            "The Python project is due tomorrow at 9 AM. "
            "Please prioritize finishing the remaining modules and push your code tonight. "
            "The client is waiting."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "easy",
        "reason": "Direct manager instruction with explicit deadline.",
    },
    {
        "text": (
            "From: support@paypa1.com\n"
            "Subject: Security Alert — Your Account Has Been Locked\n\n"
            "Your PayPal account has been locked due to suspicious activity. "
            "Click here immediately to verify your identity: http://paypa1-secure.xyz/login"
        ),
        "label": "spam",
        "priority": "low",
        "reply": "ignore",
        "tier": "easy",
        "reason": "Typo in sender domain (paypa1 not paypal) + suspicious link = phishing.",
    },
    {
        "text": (
            "From: deals@shopnow.com\n"
            "Subject: 🎉 MEGA SALE — 70% OFF Everything This Weekend Only!\n\n"
            "Don't miss our biggest sale of the year. "
            "Shop now and save big on electronics, fashion, and more. "
            "Use code SAVE70 at checkout. Limited time!"
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "easy",
        "reason": "Classic promotional email with discount codes and urgency marketing.",
    },
    {
        "text": (
            "From: hr@hiring.com\n"
            "Subject: Interview Invitation — Software Engineer Role\n\n"
            "We would like to schedule a technical interview with you for the "
            "Software Engineer position. Please confirm your availability for "
            "a call at 10 AM on Friday."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "easy",
        "reason": "Interview invitation requires direct confirmation response.",
    },
    {
        "text": (
            "From: newsletter@techdigest.io\n"
            "Subject: This Week in Tech — AI & Cloud Roundup\n\n"
            "Here is your weekly digest of top stories in AI, cloud computing, "
            "and software engineering. Stories this week: GPT-5 rumors, AWS outage, "
            "and the rise of edge computing."
        ),
        "label": "promotional",
        "priority": "medium",
        "reply": "acknowledge",
        "tier": "easy",
        "reason": "Subscribed newsletter — promotional but worth acknowledging receipt.",
    },
    {
        "text": (
            "From: no-reply@instagram.com\n"
            "Subject: You have new followers this week!\n\n"
            "Hi, 5 people followed you this week on Instagram. "
            "Check out who's following you and see their latest posts."
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "easy",
        "reason": "Social media notification — no action needed.",
    },
    {
        "text": (
            "From: ceo@mycompany.com\n"
            "Subject: All Hands Meeting — Tomorrow 2 PM\n\n"
            "Team, we are holding a mandatory all-hands meeting tomorrow at 2 PM "
            "in the main conference room. Attendance is required. "
            "We will be discussing the Q3 results and the roadmap for Q4."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "easy",
        "reason": "Mandatory meeting from CEO — must respond to confirm attendance.",
    },

    # ══════════════════════════════════════════════════════
    # TIER: MEDIUM — requires reading carefully, mild ambiguity
    # ══════════════════════════════════════════════════════

    {
        "text": (
            "From: billing@netflix.com\n"
            "Subject: Your payment of $15.99 was successful\n\n"
            "Hi, your Netflix subscription has been renewed for another month. "
            "Your next billing date is November 15. "
            "If you did not authorize this charge, contact us immediately."
        ),
        "label": "important",
        "priority": "medium",
        "reply": "acknowledge",
        "tier": "medium",
        "reason": "Legitimate billing receipt — important to keep for records, no urgent reply needed.",
    },
    {
        "text": (
            "From: alerts@github.com\n"
            "Subject: [Security] A new sign-in to your GitHub account\n\n"
            "We noticed a new sign-in to your GitHub account from a device we don't recognize. "
            "Location: Mumbai, India. If this was you, no action is needed. "
            "If not, secure your account immediately."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "medium",
        "reason": "Security alert from verified GitHub domain — requires immediate action if unauthorized.",
    },
    {
        "text": (
            "From: promo@amazon.com\n"
            "Subject: Your Amazon order #112-4857291 has shipped!\n\n"
            "Great news! Your order for 'Clean Code by Robert C. Martin' has shipped. "
            "Expected delivery: Thursday. Track your package: amazon.com/track"
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "medium",
        "reason": "Order confirmation from Amazon — informational, no reply needed.",
    },
    {
        "text": (
            "From: professor.sharma@university.edu\n"
            "Subject: Assignment 3 — Submission Deadline Extended\n\n"
            "Dear students, due to the server issues last week, "
            "I am extending the deadline for Assignment 3 to this Sunday at midnight. "
            "Please ensure you upload via the portal, not email."
        ),
        "label": "important",
        "priority": "high",
        "reply": "acknowledge",
        "tier": "medium",
        "reason": "Academic deadline change from professor — important, acknowledge receipt.",
    },
    {
        "text": (
            "From: team@producthunt.com\n"
            "Subject: 🚀 Top Products of the Week — Nov Edition\n\n"
            "This week's hottest launches on Product Hunt include a new AI writing tool, "
            "a Notion alternative, and a code review bot. "
            "Check out what the maker community is building!"
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "medium",
        "reason": "Product Hunt digest — promotional newsletter, no action required.",
    },
    {
        "text": (
            "From: friend.rahul@gmail.com\n"
            "Subject: Can you help me move this Saturday?\n\n"
            "Hey! I'm shifting apartments this Saturday and could really use an extra pair "
            "of hands. It should take about 3-4 hours max. I'll order pizza after! "
            "Let me know if you're free."
        ),
        "label": "important",
        "priority": "medium",
        "reply": "respond",
        "tier": "medium",
        "reason": "Personal request from a friend needing a direct yes/no reply.",
    },

    # ══════════════════════════════════════════════════════
    # TIER: HARD — genuinely ambiguous, tricky signals
    # ══════════════════════════════════════════════════════

    {
        "text": (
            "From: careers@google.com\n"
            "Subject: Exploring opportunities at Google\n\n"
            "Hi, I came across your profile and wanted to reach out about potential "
            "opportunities at Google that align with your background in machine learning. "
            "Would you be open to a quick 15-minute call this week?"
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "hard",
        "reason": "Cold recruiter from Google — looks promotional but is a real career opportunity.",
    },
    {
        "text": (
            "From: support@apple.com\n"
            "Subject: Your Apple ID was used to sign in on a new iPhone\n\n"
            "Your Apple ID was used to sign in to iCloud on an iPhone 15 Pro. "
            "If this is you, you can ignore this message. "
            "If you did not sign in, your account may be compromised."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "hard",
        "reason": "Looks like a phishing template but sender is legitimate apple.com — high priority security.",
    },
    {
        "text": (
            "From: noreply@linkedin.com\n"
            "Subject: You appeared in 47 searches this week\n\n"
            "Your profile is getting noticed! You appeared in 47 searches this week "
            "and 3 recruiters viewed your profile. Upgrade to Premium to see who's looking."
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "hard",
        "reason": "LinkedIn upsell notification disguised as career insight — promotional.",
    },
    {
        "text": (
            "From: payments@freelancer.com\n"
            "Subject: You have a pending payment of $240\n\n"
            "A client has released a milestone payment of $240 for the project "
            "'React Dashboard UI'. The funds will be available in your account "
            "within 3-5 business days. No action required."
        ),
        "label": "important",
        "priority": "medium",
        "reply": "acknowledge",
        "tier": "hard",
        "reason": "Payment notification — important financial record but no urgent reply needed.",
    },
    {
        "text": (
            "From: do-not-reply@coursera.org\n"
            "Subject: Your certificate is ready!\n\n"
            "Congratulations! You have successfully completed 'Machine Learning Specialization' "
            "by Stanford University. Your certificate is ready to download and share "
            "on LinkedIn."
        ),
        "label": "promotional",
        "priority": "low",
        "reply": "ignore",
        "tier": "hard",
        "reason": "Course completion notification — positive but promotional, no reply needed.",
    },
    {
        "text": (
            "From: \n"
            "Subject: \n\n"
            "Hey just checking if you got my last email. Let me know."
        ),
        "label": "important",
        "priority": "medium",
        "reply": "respond",
        "tier": "hard",
        "reason": "Edge case: missing sender/subject. Vague follow-up still warrants a response.",
    },
    {
        "text": (
            "From: internship@startup.io\n"
            "Subject: Re: Your application — next steps\n\n"
            "Hi! Thanks for applying to our summer internship program. "
            "We were really impressed with your portfolio. "
            "We'd love to move forward — are you available for a 30-minute call this week? "
            "Please reply with two time slots that work for you."
        ),
        "label": "important",
        "priority": "high",
        "reply": "respond",
        "tier": "hard",
        "reason": "Internship callback — looks like a cold email but is a real response to an application.",
    },

]


def get_dataset(tier: str = None, shuffle: bool = False) -> list:
    """
    Return the dataset, optionally filtered by tier and/or shuffled.

    Args:
        tier:    one of 'easy', 'medium', 'hard', or None for all
        shuffle: if True, randomize order (for RL training)

    Returns:
        List of email dicts
    """
    import random
    data = EMAIL_DATASET if tier is None else [e for e in EMAIL_DATASET if e["tier"] == tier]
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    return data


def dataset_stats() -> dict:
    """Return a summary of the dataset composition."""
    from collections import Counter
    labels    = Counter(e["label"]    for e in EMAIL_DATASET)
    priorities = Counter(e["priority"] for e in EMAIL_DATASET)
    replies   = Counter(e["reply"]    for e in EMAIL_DATASET)
    tiers     = Counter(e["tier"]     for e in EMAIL_DATASET)
    return {
        "total":      len(EMAIL_DATASET),
        "tiers":      dict(tiers),
        "labels":     dict(labels),
        "priorities": dict(priorities),
        "replies":    dict(replies),
    }


if __name__ == "__main__":
    stats = dataset_stats()
    print(f"Total emails : {stats['total']}")
    print(f"Tiers        : {stats['tiers']}")
    print(f"Labels       : {stats['labels']}")
    print(f"Priorities   : {stats['priorities']}")
    print(f"Replies      : {stats['replies']}")
