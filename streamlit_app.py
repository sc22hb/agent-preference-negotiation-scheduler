import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Define the 12 available slots
SLOTS_DATA = {
    'S1': {'Day': 'Mon', 'Time': 900, 'Format': 'Teams'},
    'S2': {'Day': 'Mon', 'Time': 1030, 'Format': 'F2F'},
    'S3': {'Day': 'Mon', 'Time': 1300, 'Format': 'F2F'},
    'S4': {'Day': 'Mon', 'Time': 1430, 'Format': 'Teams'},
    'S5': {'Day': 'Mon', 'Time': 1600, 'Format': 'F2F'},
    'S6': {'Day': 'Mon', 'Time': 1700, 'Format': 'Teams'},
    
    'S7': {'Day': 'Tue', 'Time': 900, 'Format': 'F2F'},
    'S8': {'Day': 'Tue', 'Time': 1030, 'Format': 'Teams'},
    'S9': {'Day': 'Tue', 'Time': 1300, 'Format': 'Teams'},
    'S10': {'Day': 'Tue', 'Time': 1430, 'Format': 'F2F'},
    'S11': {'Day': 'Tue', 'Time': 1600, 'Format': 'Teams'},
    'S12': {'Day': 'Tue', 'Time': 1700, 'Format': 'F2F'},
}
SLOTS_DF = pd.DataFrame.from_dict(SLOTS_DATA, orient='index')
SLOTS_DF['Time_Display'] = SLOTS_DF['Time'].apply(lambda t: f"{t//100:02d}:{t%100:02d}")
SLOT_IDS = list(SLOTS_DATA.keys())

# Define the 8 people (the Agents)
PERSON_IDS = [f'Person {i+1}' for i in range(8)]
VALID_DAYS = ['Mon', 'Tue'] 

# Define the scoring weights
WEIGHTS = {
    'Yes': 3,
    'No': -5,  # Strong negative to act as a near-disqualifier
    'Dont Care': 0,
    'Time_Pref': 2,
    'Time_Avoid': -4
}

# Sentinel for slots that violate hard constraints (e.g., "Definitely NOT" day)
FORBIDDEN_SCORE = -9999

# Preference strength heuristic: counts how specific/expressive the user is.
def preference_strength(prefs):
    strength = 0
    # Day preference: anything other than neutral earns strength
    if prefs.get('Day_Preference') not in ["Don't Care", "Either (Mon or Tue) is fine"]:
        strength += 1
    # Format preferences: non-neutral responses add strength
    for f in ['F2F', 'Teams']:
        if prefs.get(f) != 'Dont Care':
            strength += 1
    # Time preference: anything other than 'Anytime' adds strength
    if prefs.get('Time') != 'Anytime':
        strength += 1
    return strength

# --- TIME CHOICES ---
TIME_CHOICES = {
    'Anytime': (0, 2400),
    'Before 10 AM': (0, 1000),
    'Before 1 PM (13:00)': (0, 1300),
    'After 12 PM': (1200, 2400),
    'After 4 PM (16:00)': (1600, 2400), 
    'Between 10 AM and 4 PM': (1000, 1600),
}


# --- Combined Day Preference Options ---
DAY_PREF_OPTIONS = [
    'Don\'t Care',
    'Prefer Monday',
    'Prefer Tuesday',
    'Either (Mon or Tue) is fine', # Neutral/Don't Care, but explicit
    'Definitely NOT Monday',
    'Definitely NOT Tuesday'
]



def inject_base_styles():
    """Lightweight CSS to tidy spacing and add subtle modern styling."""
    st.markdown(
        """
        <style>
            /* Global layout */
            .app-container {
                background: radial-gradient(circle at top left, #eff6ff 0, #f8fafc 40%, #f9fafb 100%);
                padding: 0 1.5rem 2.5rem 1.5rem;
            }

            .block-container {
                max-width: 1100px !important;
                padding-top: 1.5rem;
            }

            /* Typography */
            h1, h2, h3, h4, h5 {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
                color: #0f172a;
            }

            .app-subtitle {
                color: #475569;
                font-size: 0.95rem;
                margin-top: -0.35rem;
                margin-bottom: 0.75rem;
            }

            /* Cards & panels */
            .card {
                background: #ffffffdd;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
                backdrop-filter: blur(6px);
            }

            .card + .card {
                margin-top: 0.9rem;
            }

            .section-title {
                margin-top: 0px;
                margin-bottom: 0.25rem;
            }

            /* Metrics row */
            .metric-row {
                margin-top: 0.25rem;
                margin-bottom: 0.75rem;
            }

            /* DataFrames */
            .dataframe td, .dataframe th {
                border: none !important;
            }
            .dataframe th {
                background-color: #eff6ff !important;
                color: #1e293b !important;
                font-weight: 600 !important;
            }

            /* Buttons */
            .stButton>button {
                border-radius: 999px;
                background: linear-gradient(90deg, #2563eb, #4f46e5);
                color: #ffffff;
                border: none;
                padding: 0.4rem 1.4rem;
                font-weight: 600;
                box-shadow: 0 10px 25px rgba(37, 99, 235, 0.35);
            }

            .stButton>button:hover {
                background: linear-gradient(90deg, #1d4ed8, #4338ca);
                box-shadow: 0 14px 30px rgba(37, 99, 235, 0.45);
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: #0f172a;
            }
            [data-testid="stSidebar"] * {
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] h2 {
                color: #e5e7eb !important;
            }

            /* Tabs */
            button[role="tab"] {
                border-radius: 999px !important;
            }

            /* Small utility labels */
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
                border-radius: 999px;
                padding: 0.1rem 0.6rem;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                background: #e0f2fe;
                color: #0369a1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- 3. AGENT LOGIC (SCORING) ---

def calculate_utility_score(slot_id, slot_data, prefs):
    """Calculates the P-Agent's Utility Score for a specific slot."""
    score = 0
    slot_day = slot_data['Day']
    slot_time = slot_data['Time']
    slot_format = slot_data['Format']
    
    # --- Day Preference ---
    day_pref = prefs['Day_Preference']

    # Hard exclusions: "Definitely NOT" days are treated as forbidden.
    if day_pref == 'Definitely NOT Monday' and slot_day == 'Mon':
        return FORBIDDEN_SCORE
    if day_pref == 'Definitely NOT Tuesday' and slot_day == 'Tue':
        return FORBIDDEN_SCORE
    
    if day_pref == 'Prefer Monday':
        if slot_day == 'Mon':
            score += WEIGHTS['Yes'] # +3 for preferred day
    elif day_pref == 'Prefer Tuesday':
        if slot_day == 'Tue':
            score += WEIGHTS['Yes'] # +3 for preferred day
    elif day_pref == 'Definitely NOT Monday':
        if slot_day == 'Mon':
            score += WEIGHTS['No'] # -5 for hard avoid
    elif day_pref == 'Definitely NOT Tuesday':
        if slot_day == 'Tue':
            score += WEIGHTS['No'] # -5 for hard avoid
    
    # 'Don\'t Care' and 'Either (Mon or Tue) add 0, as they are neutral.

    # --- Format Preference (F2F and Teams) ---
    for f in ['F2F', 'Teams']:
        pref_f = prefs[f]
        if pref_f == 'Yes' and slot_format == f:
            score += WEIGHTS['Yes']
        elif pref_f == 'No' and slot_format == f:
            score += WEIGHTS['No']
        # 'Dont Care' adds 0

    # --- Time Preference ---
    time_key = prefs['Time']
    lower, upper = TIME_CHOICES[time_key]
    
    # Check if slot time falls within preferred range (positive score)
    if lower <= slot_time < upper and time_key != 'Anytime':
        score += WEIGHTS['Time_Pref']
    
    return score

# --- 4. MASTER AGENT LOGIC (NEGOTIATION) ---

def master_agent_negotiate(all_person_prefs, render_log=False):
    """Runs the auction-style assignment process."""
    available_slots = set(SLOT_IDS)
    assigned_slots = {}  # {Person_ID: Slot_ID}
    
    log_messages = []
    
    # Run a maximum of 5 rounds (or until all 8 people are assigned)
    for round_num in range(1, 6):
        if len(assigned_slots) == len(PERSON_IDS):
            log_messages.append(f"✅ All {len(PERSON_IDS)} people assigned! Negotiation complete.")
            break

        if render_log:
            st.subheader(f"Round {round_num}")
        round_bids = defaultdict(list) # {Slot_ID: [(score, Person_ID)]}
        unassigned_people = [p for p in PERSON_IDS if p not in assigned_slots]
        
        # 1. Collect Bids from P-Agents
        for person_id in unassigned_people:
            prefs = all_person_prefs[person_id]
            
            # P-Agent scores all available slots
            scores = {}
            for slot_id in available_slots:
                slot_data = SLOTS_DATA[slot_id]
                score = calculate_utility_score(slot_id, slot_data, prefs)
                scores[slot_id] = score
            
            # P-Agent determines its top bid in three tiers:
            # 1) Any non-negative slots (best effort)
            positive_slots = {s: sc for s, sc in scores.items() if sc >= 0 and sc > FORBIDDEN_SCORE}
            # 2) Slightly negative but still acceptable (>-4), never forbidden
            mild_negative_slots = {s: sc for s, sc in scores.items() if -4 < sc < 0 and sc > FORBIDDEN_SCORE}
            # 3) Last-resort: best non-forbidden slot, even if negative (keeps the person assignable)
            fallback_slots = {s: sc for s, sc in scores.items() if sc > FORBIDDEN_SCORE}

            eligible_slots = None
            if positive_slots:
                eligible_slots = positive_slots
            elif mild_negative_slots:
                eligible_slots = mild_negative_slots
            elif fallback_slots:
                eligible_slots = fallback_slots

            if not eligible_slots:
                log_messages.append(f"⚠️ **{person_id}** has no eligible slots remaining.")
                continue

            # Bid on the highest scoring slot; if multiple tie (e.g., all Don't Care),
            # pick randomly among the best to avoid everyone piling onto one slot.
            best_score = max(eligible_slots.values())
            best_candidates = [s for s, sc in eligible_slots.items() if sc == best_score]
            best_slot = random.choice(best_candidates)
            
            round_bids[best_slot].append((best_score, person_id))
            log_messages.append(f"**{person_id}** bids on **{best_slot}** with score **{best_score}**.")

        # 2. Resolve Conflicts and Make Provisional Assignments
        round_assignments = {} 
        
        for slot_id, bids in round_bids.items():
            if len(bids) == 1:
                # No conflict - Assign the slot
                score, person_id = bids[0]
                round_assignments[slot_id] = (person_id, score)
                log_messages.append(f"🟢 **{slot_id}** assigned to **{person_id}** (Score: {score}). No conflict.")
            else:
                # Conflict - Assign to the highest utility score; break ties by preference strength, then random
                def bid_key(bid):
                    sc, pid = bid
                    return (sc, preference_strength(all_person_prefs[pid]))
                top_score = max(bids, key=bid_key)[0]
                top_bids = [b for b in bids if b[0] == top_score]
                if len(top_bids) > 1:
                    # tie on score; prefer higher strength
                    top_strength = max(preference_strength(all_person_prefs[b[1]]) for b in top_bids)
                    strength_bids = [b for b in top_bids if preference_strength(all_person_prefs[b[1]]) == top_strength]
                    winning_bid = random.choice(strength_bids)
                else:
                    winning_bid = top_bids[0]

                score, person_id = winning_bid
                round_assignments[slot_id] = (person_id, score)
                
                # Log who was rejected
                rejected = [p for s, p in bids if p != person_id]
                log_messages.append(f"🔴 **{slot_id}** conflict. Assigned to **{person_id}** (Score: {score}). Rejected: {', '.join(rejected)}")

        # 3. Update Global State
        newly_assigned_count = 0
        for slot_id, (person_id, score) in round_assignments.items():
            if person_id not in assigned_slots:
                assigned_slots[person_id] = slot_id
                available_slots.remove(slot_id)
                newly_assigned_count += 1
        
        if newly_assigned_count == 0 and len(unassigned_people) > 0:
            log_messages.append("🚫 No new assignments made this round. Ending negotiation early.")
            break
        
        if render_log:
            st.info(f"Assignments made in Round {round_num}: **{newly_assigned_count}**")
        
    if render_log:
        st.text("\n".join(log_messages))
    return assigned_slots, log_messages

# --- 5. STREAMLIT APP INTERFACE ---

def main():
    st.set_page_config(layout="wide", page_title="Agent-Based Appointment Scheduler")
    inject_base_styles()
    st.markdown('<div class="app-container">', unsafe_allow_html=True)

    
    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.title("Multi Agent-Based Appointment Scheduler")
    with header_col2:
        with st.container():
            st.markdown('<div class="card metric-row">', unsafe_allow_html=True)
            cols = st.columns(3)
            cols[0].metric("People", len(PERSON_IDS))
            cols[1].metric("Slots", len(SLOT_IDS))
            cols[2].metric("Max Rounds", 5)
            st.markdown("</div>", unsafe_allow_html=True)

    # Display Slots Table
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Available Slots (Inventory: 12)")
        display_df = SLOTS_DF.copy()
        display_df = display_df.drop(columns=['Time'])
        display_df.index.name = 'Slot ID'
        st.dataframe(display_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.header("Agent Preferences Input")
    st.sidebar.markdown(
        "Define how each person feels about **day**, **format** and **time of day**. "
        "When you are ready, run the negotiation to generate assignments.",
    )
    
    # Initialize session state for storing preferences
    if 'preferences' not in st.session_state:
        # Set default preferences for demonstration
        default_prefs = {}
        for p in PERSON_IDS:
            default_prefs[p] = {
                'Day_Preference': 'Don\'t Care',
                'F2F': 'Dont Care',
                'Teams': 'Dont Care',
                'Time': 'Anytime',
            }
        st.session_state.preferences = default_prefs

    # Create one form for all 8 people
    with st.form("preference_form"):
        st.markdown("### Set Preferences for Each Person")
        
        cols = st.columns(4)
        
        # Helper function for input key
        def get_key(person, attr):
            return f"{person}_{attr}"

        for i, person_id in enumerate(PERSON_IDS):
            with cols[i % 4]:
                st.markdown(f"#### {person_id}")
                
                # Combined Day Preference Input
                day_pref = st.selectbox(
                    f"Day Preference",
                    DAY_PREF_OPTIONS,
                    index=DAY_PREF_OPTIONS.index(st.session_state.preferences[person_id]['Day_Preference']),
                    key=get_key(person_id, 'Day_Preference')
                )
                
                # Format Preference
                f2f = st.radio(
                    f"Face-to-Face",
                    ['Dont Care', 'Yes', 'No'],
                    index=['Dont Care', 'Yes', 'No'].index(st.session_state.preferences[person_id]['F2F']),
                    key=get_key(person_id, 'F2F')
                )
                teams = st.radio(
                    f"Teams",
                    ['Dont Care', 'Yes', 'No'],
                    index=['Dont Care', 'Yes', 'No'].index(st.session_state.preferences[person_id]['Teams']),
                    key=get_key(person_id, 'Teams')
                )

                # Time Preferences
                time_pref = st.selectbox(
                    f"Time of Day",
                    list(TIME_CHOICES.keys()),
                    index=list(TIME_CHOICES.keys()).index(st.session_state.preferences[person_id]['Time']),
                    key=get_key(person_id, 'Time')
                )
                
                # Update session state with form values
                st.session_state.preferences[person_id] = {
                    'Day_Preference': day_pref,
                    'F2F': f2f,
                    'Teams': teams,
                    'Time': time_pref,
                }
        
        st.markdown("---")
        submitted = st.form_submit_button("Run Master Agent Negotiation")

    if submitted:
        st.sidebar.subheader("Input Review")
        st.sidebar.dataframe(pd.DataFrame(st.session_state.preferences).T)

        final_assignments, log_messages = master_agent_negotiate(st.session_state.preferences, render_log=False)
        
        st.header("Final Assignment Results")
        results_tab, log_tab, inputs_tab = st.tabs(["Results", "Negotiation Log", "Inputs Review"])

        with results_tab:
            if final_assignments:
                results_list = []
                for person, slot_id in final_assignments.items():
                    slot_info = SLOTS_DATA[slot_id]
                    final_score = calculate_utility_score(slot_id, slot_info, st.session_state.preferences[person])
                    
                    results_list.append({
                        'Person': person,
                        'Assigned Slot': slot_id,
                        'Day': slot_info['Day'],
                        'Time': SLOTS_DF.loc[slot_id, 'Time_Display'],
                        'Format': slot_info['Format'],
                        'Utility Score': final_score,
                    })

                results_df = pd.DataFrame(results_list)
                results_df = results_df.sort_values(by='Utility Score', ascending=False).reset_index(drop=True)
                
                st.success(f"Successfully assigned **{len(results_df)}** out of 8 people to slots!")
                st.dataframe(results_df, use_container_width=True)
                
                st.markdown("### Unassigned Slots")
                assigned_slot_ids = set(final_assignments.values())
                unassigned_slots = SLOTS_DF[~SLOTS_DF.index.isin(assigned_slot_ids)].drop(columns=['Time'])
                st.dataframe(unassigned_slots, use_container_width=True)
            else:
                st.error("The Master Agent failed to make any assignments.")

        with log_tab:
            if log_messages:
                st.code("\n".join(log_messages))
            else:
                st.info("No log messages to display.")

        with inputs_tab:
            st.dataframe(pd.DataFrame(st.session_state.preferences).T)


if __name__ == "__main__":
    main()