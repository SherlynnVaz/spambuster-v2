# ── Short Spam Augmentation ───────────────────────────────────
# Adds short, ambiguous spam messages to the training data
# to help BERT learn spam patterns without long context

import pandas as pd

print("Adding short spam examples...")

# ── SHORT SPAM EXAMPLES ───────────────────────────────────────
short_spam = [
    # Prize/winning
    "Free lotto", "You won!", "Claim prize", "Winner selected",
    "Free prize", "You have won", "Claim now", "Prize winner",
    "Lucky winner", "Win cash", "Free gift", "Claim reward",
    "You are winner", "Collect prize", "Free money",
    "Congratulations winner", "Prize waiting", "Claim free gift",
    "Win today", "Free reward", "You qualify", "Claim yours",

    # Urgency
    "Act now", "Limited time", "Expires tonight", "Last chance",
    "Urgent offer", "Today only", "Reply immediately",
    "Don't miss out", "Offer expires", "Final notice",
    "Respond now", "Time running out", "Hurry limited offer",

    # Free stuff
    "Free iPhone", "Free Samsung", "Free voucher", "Free trial",
    "Free subscription", "Totally free", "100% free",
    "Free access", "Get free", "Free today only",

    # Account/security phishing
    "Verify account", "Account suspended", "Login required",
    "Confirm identity", "Account locked", "Update details",
    "Verify now", "Account alert", "Security update required",
    "Confirm your details", "Account compromised",

    # Money
    "Easy money", "Make money fast", "Earn cash now",
    "Get paid today", "Quick cash", "Instant money",
    "Cash reward waiting", "Earn from home", "Double your money",

    # Call to action
    "Call now", "Click here", "Reply YES", "Text back",
    "Call free", "SMS back", "Reply now", "Click link",
    "Visit now", "Register free", "Sign up free",

    # Mixed
    "WINNER!! Free prize", "FREE gift claim now",
    "Urgent free reward", "Win free iPhone today",
    "Claim your free cash", "You won free gift",
    "Free lotto winner", "WINNER claim prize",
    "Free money waiting", "Urgent prize claim",
    "Win big today", "Cash prize winner",
    "Free entry winner", "Claim free reward now",
    "You qualify free prize", "Urgent account verify",
    "FREE winner selected", "Congratulations free gift",
    "Limited free offer", "Win free cash today",
]

# ── SHORT HAM EXAMPLES ────────────────────────────────────────
# Also add short ham so model learns balance
short_ham = [
    "Ok sounds good", "See you then", "Thanks a lot",
    "Sure no problem", "On my way", "Be there soon",
    "Running late sorry", "Call me later", "Miss you",
    "Good morning", "How are you", "Thinking of you",
    "Happy birthday", "Congratulations on your promotion",
    "Good luck today", "Take care", "Drive safe",
    "Let me know", "Sounds like a plan", "I agree",
    "Talk later", "Will do", "Got it thanks",
    "No problem at all", "See you tomorrow",
    "Have a good day", "Hope you feel better",
    "Just checking in", "All good here", "Doing well thanks",
    "Free this evening", "Free for lunch tomorrow",
    "Are you free", "Free to talk now",
    "Congratulations on the baby", "Congrats on the new job",
    "Happy anniversary", "Happy new year",
    "Merry Christmas", "Hope you had fun",
]

# ── BUILD DATAFRAME ───────────────────────────────────────────
spam_df = pd.DataFrame({
    'label': ['spam'] * len(short_spam),
    'message': short_spam
})

ham_df = pd.DataFrame({
    'label': ['ham'] * len(short_ham),
    'message': short_ham
})

new_data = pd.concat([spam_df, ham_df], ignore_index=True)

print(f"New short examples: {len(new_data)}")
print(f"  Spam: {len(spam_df)}")
print(f"  Ham:  {len(ham_df)}")

# ── LOAD EXISTING MERGED DATASET ─────────────────────────────
existing = pd.read_csv('training/merged_dataset.csv')
print(f"\nExisting dataset: {len(existing)}")

# ── MERGE ────────────────────────────────────────────────────
combined = pd.concat([existing, new_data], ignore_index=True)
combined = combined.drop_duplicates(subset=['message'])
combined = combined.reset_index(drop=True)

print(f"Combined dataset: {len(combined)}")
print(f"  Spam: {len(combined[combined.label=='spam'])}")
print(f"  Ham:  {len(combined[combined.label=='ham'])}")

# ── SAVE ─────────────────────────────────────────────────────
combined.to_csv('training/merged_dataset.csv', index=False)
print(f"\n✅ Saved to training/merged_dataset.csv")