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
SLIDER_WEIGHTS = {
    'Day': 0.6,
    'Format': 0.7,
    'Time': 0.5,
}

CATEGORICAL_WEIGHTS = {
    'Yes': 3,
    'No': -5,
    'Dont Care': 0,
    'Time_Pref': 2,
}

# Normalization targets for report-style outputs
SLIDER_MAX_SCORE = 100 * (SLIDER_WEIGHTS['Day'] + SLIDER_WEIGHTS['Format'] + SLIDER_WEIGHTS['Time'])
CATEGORICAL_MIN_SCORE = -5
CATEGORICAL_MAX_SCORE = 8

# Slot time scaling for 0-100 preference alignment
MIN_SLOT_TIME = min(slot['Time'] for slot in SLOTS_DATA.values())
MAX_SLOT_TIME = max(slot['Time'] for slot in SLOTS_DATA.values())

# --- TIME CHOICES (Categorical) ---
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
    'Either (Mon or Tue) is fine',
    'Definitely NOT Monday',
    'Definitely NOT Tuesday'
]

FORBIDDEN_SCORE = -9999

# Preference strength heuristic: how far from neutral (50) the user is.
def preference_strength_slider(prefs):
    strength = 0.0
    for key in ['Pref_Mon', 'Pref_Tue', 'Pref_F2F', 'Pref_Teams', 'Pref_Time']:
        strength += abs(prefs.get(key, 50) - 50) / 50
    return strength

def preference_strength_categorical(prefs):
    strength = 0
    if prefs.get('Day_Preference') not in ["Don't Care", "Either (Mon or Tue) is fine"]:
        strength += 1
    for f in ['F2F', 'Teams']:
        if prefs.get(f) != 'Dont Care':
            strength += 1
    if prefs.get('Time') != 'Anytime':
        strength += 1
    return strength

def slot_time_to_scale(slot_time):
    if MAX_SLOT_TIME == MIN_SLOT_TIME:
        return 50
    return int(round((slot_time - MIN_SLOT_TIME) / (MAX_SLOT_TIME - MIN_SLOT_TIME) * 100))

def time_choice_from_slider(pref_time):
    best_choice = None
    best_dist = None
    for label, (lower, upper) in TIME_CHOICES.items():
        if label == 'Anytime':
            midpoint = 50
        else:
            midpoint_time = (lower + upper) / 2
            midpoint = slot_time_to_scale(midpoint_time)
        dist = abs(pref_time - midpoint)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_choice = label
    return best_choice

def slider_to_categorical(slider_prefs):
    pref_mon = slider_prefs['Pref_Mon']
    pref_tue = slider_prefs['Pref_Tue']
    pref_f2f = slider_prefs['Pref_F2F']
    pref_teams = slider_prefs['Pref_Teams']
    pref_time = slider_prefs['Pref_Time']

    if pref_mon <= 20 and pref_tue >= 50:
        day_pref = 'Definitely NOT Monday'
    elif pref_tue <= 20 and pref_mon >= 50:
        day_pref = 'Definitely NOT Tuesday'
    else:
        delta = pref_mon - pref_tue
        if delta >= 20:
            day_pref = 'Prefer Monday'
        elif delta <= -20:
            day_pref = 'Prefer Tuesday'
        elif abs(delta) <= 10:
            day_pref = 'Either (Mon or Tue) is fine'
        else:
            day_pref = 'Don\'t Care'

    def map_format(score):
        if score >= 70:
            return 'Yes'
        if score <= 30:
            return 'No'
        return 'Dont Care'

    return {
        'Day_Preference': day_pref,
        'F2F': map_format(pref_f2f),
        'Teams': map_format(pref_teams),
        'Time': time_choice_from_slider(pref_time),
    }

def normalize_slider_score(score):
    if SLIDER_MAX_SCORE == 0:
        return 0.0
    return max(0.0, min(100.0, (score / SLIDER_MAX_SCORE) * 100))

def normalize_categorical_score(score):
    if CATEGORICAL_MAX_SCORE == CATEGORICAL_MIN_SCORE:
        return 0.0
    return max(0.0, min(100.0, ((score - CATEGORICAL_MIN_SCORE) / (CATEGORICAL_MAX_SCORE - CATEGORICAL_MIN_SCORE)) * 100))



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

def calculate_utility_score_slider(slot_id, slot_data, prefs):
    """Calculates the P-Agent's Utility Score for a specific slot (0-100 sliders)."""
    slot_day = slot_data['Day']
    slot_time = slot_data['Time']
    slot_format = slot_data['Format']

    day_score = prefs['Pref_Mon'] if slot_day == 'Mon' else prefs['Pref_Tue']
    format_score = prefs['Pref_F2F'] if slot_format == 'F2F' else prefs['Pref_Teams']
    slot_scaled = slot_time_to_scale(slot_time)
    time_score = 100 - abs(prefs['Pref_Time'] - slot_scaled)

    score = (
        day_score * SLIDER_WEIGHTS['Day'] +
        format_score * SLIDER_WEIGHTS['Format'] +
        time_score * SLIDER_WEIGHTS['Time']
    )
    return score

def calculate_utility_score_categorical(slot_id, slot_data, prefs):
    """Calculates the P-Agent's Utility Score for a specific slot (categorical)."""
    score = 0
    slot_day = slot_data['Day']
    slot_time = slot_data['Time']
    slot_format = slot_data['Format']

    day_pref = prefs['Day_Preference']
    if day_pref == 'Definitely NOT Monday' and slot_day == 'Mon':
        return FORBIDDEN_SCORE
    if day_pref == 'Definitely NOT Tuesday' and slot_day == 'Tue':
        return FORBIDDEN_SCORE

    if day_pref == 'Prefer Monday' and slot_day == 'Mon':
        score += CATEGORICAL_WEIGHTS['Yes']
    elif day_pref == 'Prefer Tuesday' and slot_day == 'Tue':
        score += CATEGORICAL_WEIGHTS['Yes']
    elif day_pref == 'Definitely NOT Monday' and slot_day == 'Mon':
        score += CATEGORICAL_WEIGHTS['No']
    elif day_pref == 'Definitely NOT Tuesday' and slot_day == 'Tue':
        score += CATEGORICAL_WEIGHTS['No']

    for f in ['F2F', 'Teams']:
        pref_f = prefs[f]
        if pref_f == 'Yes' and slot_format == f:
            score += CATEGORICAL_WEIGHTS['Yes']
        elif pref_f == 'No' and slot_format == f:
            score += CATEGORICAL_WEIGHTS['No']

    time_key = prefs['Time']
    lower, upper = TIME_CHOICES[time_key]
    if lower <= slot_time < upper and time_key != 'Anytime':
        score += CATEGORICAL_WEIGHTS['Time_Pref']

    return score

def satisfaction_from_slider_prefs(slot_id, slider_prefs):
    slot_data = SLOTS_DATA[slot_id]
    return calculate_utility_score_slider(slot_id, slot_data, slider_prefs)

def build_results_df(assignments, prefs, mode):
    results_list = []
    for person, slot_id in assignments.items():
        slot_info = SLOTS_DATA[slot_id]
        if mode == "slider":
            raw_score = calculate_utility_score_slider(slot_id, slot_info, prefs[person])
            norm_score = normalize_slider_score(raw_score)
        else:
            raw_score = calculate_utility_score_categorical(slot_id, slot_info, prefs[person])
            norm_score = normalize_categorical_score(raw_score)

        if norm_score >= 80:
            fit_label = "Strong match"
        elif norm_score >= 60:
            fit_label = "Good match"
        elif norm_score >= 40:
            fit_label = "Mixed"
        else:
            fit_label = "Poor match"

        results_list.append({
            'Person': person,
            'Assigned Slot': slot_id,
            'Slot (Day/Time/Format)': f"{slot_info['Day']} {SLOTS_DF.loc[slot_id, 'Time_Display']} · {slot_info['Format']}",
            'Utility Score': raw_score,
            'Satisfaction (0-100)': norm_score,
            'Fit': fit_label,
        })

    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Satisfaction (0-100)', ascending=False).reset_index(drop=True)
        results_df.insert(0, 'Rank', results_df.index + 1)
    return results_df

def summarize_results(results_df):
    if results_df.empty:
        return {
            'avg': 0.0,
            'min': 0.0,
            'low_pct': 0.0,
        }
    avg_score = float(results_df['Satisfaction (0-100)'].mean())
    min_score = float(results_df['Satisfaction (0-100)'].min())
    low_pct = float((results_df['Satisfaction (0-100)'] < 50).mean() * 100)
    return {
        'avg': avg_score,
        'min': min_score,
        'low_pct': low_pct,
    }

def render_report_section(results_df, unassigned_slots_count, title_suffix):
    summary = summarize_results(results_df)
    st.subheader(f"Report Snapshot {title_suffix}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg Satisfaction", f"{summary['avg']:.1f}")
    metric_cols[1].metric("Min Satisfaction", f"{summary['min']:.1f}")
    metric_cols[2].metric("% Below 50", f"{summary['low_pct']:.1f}%")
    metric_cols[3].metric("Unassigned Slots", f"{unassigned_slots_count}")

    st.markdown(
        "How to read this: **Satisfaction (0-100)** is normalized for reporting. "
        "Fit labels are based on that score (80+ strong, 60-79 good, 40-59 mixed, <40 poor)."
    )

    if not results_df.empty:
        chart_df = results_df[['Person', 'Satisfaction (0-100)']].set_index('Person')
        st.bar_chart(chart_df)

# --- 4. MASTER AGENT LOGIC (NEGOTIATION) ---

def master_agent_negotiate(all_person_prefs, scorer, strength_fn, render_log=False, person_ids=None):
    """Runs the auction-style assignment process."""
    if person_ids is None:
        person_ids = PERSON_IDS
    available_slots = set(SLOT_IDS)
    assigned_slots = {}  # {Person_ID: Slot_ID}
    
    log_messages = []
    
    # Run a maximum of 5 rounds (or until all 8 people are assigned)
    for round_num in range(1, 6):
        if len(assigned_slots) == len(person_ids):
            log_messages.append(f"✅ All {len(person_ids)} people assigned! Negotiation complete.")
            break

        if render_log:
            st.subheader(f"Round {round_num}")
        round_bids = defaultdict(list) # {Slot_ID: [(score, Person_ID)]}
        unassigned_people = [p for p in person_ids if p not in assigned_slots]
        
        # 1. Collect Bids from P-Agents
        for person_id in unassigned_people:
            prefs = all_person_prefs[person_id]
            
            # P-Agent scores all available slots
            scores = {}
            for slot_id in available_slots:
                slot_data = SLOTS_DATA[slot_id]
                score = scorer(slot_id, slot_data, prefs)
                scores[slot_id] = score
            
            eligible_slots = {s: sc for s, sc in scores.items() if sc > FORBIDDEN_SCORE}
            if not eligible_slots:
                log_messages.append(f"⚠️ **{person_id}** has no eligible slots remaining.")
                continue

            # Bid on the highest scoring slot; if multiple tie (e.g., all 50s),
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
                    return (sc, strength_fn(all_person_prefs[pid]))
                top_score = max(bids, key=bid_key)[0]
                top_bids = [b for b in bids if b[0] == top_score]
                if len(top_bids) > 1:
                    # tie on score; prefer higher strength
                    top_strength = max(strength_fn(all_person_prefs[b[1]]) for b in top_bids)
                    strength_bids = [b for b in top_bids if strength_fn(all_person_prefs[b[1]]) == top_strength]
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

def simulate_other_bookings_random(scorer, strength_fn, target_person, num_other_bookings, mode_label):
    """Simulate other agents booking slots before the target user using random prefs."""
    other_people = [p for p in PERSON_IDS if p != target_person]
    if num_other_bookings > len(other_people):
        num_other_bookings = len(other_people)
    selected_others = random.sample(other_people, num_other_bookings)

    if mode_label == "slider":
        other_prefs = {
            p: {
                'Pref_Mon': random.randint(0, 100),
                'Pref_Tue': random.randint(0, 100),
                'Pref_F2F': random.randint(0, 100),
                'Pref_Teams': random.randint(0, 100),
                'Pref_Time': random.randint(0, 100),
            }
            for p in selected_others
        }
    else:
        other_prefs = {}
        for p in selected_others:
            sp = {
                'Pref_Mon': random.randint(0, 100),
                'Pref_Tue': random.randint(0, 100),
                'Pref_F2F': random.randint(0, 100),
                'Pref_Teams': random.randint(0, 100),
                'Pref_Time': random.randint(0, 100),
            }
            other_prefs[p] = slider_to_categorical(sp)

    other_assignments, _ = master_agent_negotiate(
        other_prefs,
        scorer=scorer,
        strength_fn=strength_fn,
        render_log=False,
        person_ids=selected_others
    )
    occupied_slots = set(other_assignments.values())
    return other_assignments, occupied_slots, selected_others

def assign_single_user(prefs, target_person, occupied_slots, scorer):
    """Assign best available slot to a single user."""
    available_slots = [s for s in SLOT_IDS if s not in occupied_slots]
    if not available_slots:
        return None
    scores = {s: scorer(s, SLOTS_DATA[s], prefs[target_person]) for s in available_slots}
    best_score = max(scores.values())
    best_candidates = [s for s, sc in scores.items() if sc == best_score]
    return random.choice(best_candidates)

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
    app_mode = st.sidebar.radio(
        "App Mode",
        ["Multi-user negotiation", "Single-user booking (simulate others)"],
        index=0
    )
    
    # Initialize session state for storing preferences
    if 'slider_prefs' not in st.session_state:
        default_prefs = {}
        for p in PERSON_IDS:
            default_prefs[p] = {
                'Pref_Mon': 50,
                'Pref_Tue': 50,
                'Pref_F2F': 50,
                'Pref_Teams': 50,
                'Pref_Time': 50,
            }
        st.session_state.slider_prefs = default_prefs

    if 'categorical_prefs' not in st.session_state:
        default_prefs = {}
        for p in PERSON_IDS:
            default_prefs[p] = {
                'Day_Preference': 'Don\'t Care',
                'F2F': 'Dont Care',
                'Teams': 'Dont Care',
                'Time': 'Anytime',
            }
        st.session_state.categorical_prefs = default_prefs

    input_mode = st.sidebar.radio(
        "Preference Input Mode",
        ["Sliders (0-100)", "Hardcoded (Yes/No)", "Both"],
        index=0
    )

    target_person = None
    other_bookings = 0
    if app_mode == "Single-user booking (simulate others)":
        target_person = st.sidebar.selectbox("Target user", PERSON_IDS, index=0)
        other_bookings = st.sidebar.slider("Number of other bookings to simulate", 0, len(PERSON_IDS) - 1, 5)

    # Create one form for all 8 people
    with st.form("preference_form"):
        st.markdown("### Set Preferences for Each Person")
        
        cols = st.columns(4)
        
        # Helper function for input key
        def get_key(person, attr):
            return f"{person}_{attr}"

        people_to_render = PERSON_IDS if app_mode == "Multi-user negotiation" else [target_person]
        for i, person_id in enumerate(people_to_render):
            with cols[i % 4]:
                st.markdown(f"#### {person_id}")
                
                if input_mode in ["Sliders (0-100)", "Both"]:
                    st.markdown("**Sliders (0-100)**")
                    pref_mon = st.slider(
                        "Prefer Monday",
                        0,
                        100,
                        st.session_state.slider_prefs[person_id]['Pref_Mon'],
                        key=get_key(person_id, 'Pref_Mon')
                    )
                    pref_tue = st.slider(
                        "Prefer Tuesday",
                        0,
                        100,
                        st.session_state.slider_prefs[person_id]['Pref_Tue'],
                        key=get_key(person_id, 'Pref_Tue')
                    )
                    pref_f2f = st.slider(
                        "Prefer Face-to-Face",
                        0,
                        100,
                        st.session_state.slider_prefs[person_id]['Pref_F2F'],
                        key=get_key(person_id, 'Pref_F2F')
                    )
                    pref_teams = st.slider(
                        "Prefer Teams",
                        0,
                        100,
                        st.session_state.slider_prefs[person_id]['Pref_Teams'],
                        key=get_key(person_id, 'Pref_Teams')
                    )
                    pref_time = st.slider(
                        "Preferred Time of Day (Earlier → Later)",
                        0,
                        100,
                        st.session_state.slider_prefs[person_id]['Pref_Time'],
                        key=get_key(person_id, 'Pref_Time')
                    )
                    st.session_state.slider_prefs[person_id] = {
                        'Pref_Mon': pref_mon,
                        'Pref_Tue': pref_tue,
                        'Pref_F2F': pref_f2f,
                        'Pref_Teams': pref_teams,
                        'Pref_Time': pref_time,
                    }

                if input_mode in ["Hardcoded (Yes/No)", "Both"]:
                    st.markdown("**Hardcoded**")
                    day_pref = st.selectbox(
                        f"Day Preference",
                        DAY_PREF_OPTIONS,
                        index=DAY_PREF_OPTIONS.index(st.session_state.categorical_prefs[person_id]['Day_Preference']),
                        key=get_key(person_id, 'Day_Preference')
                    )
                    f2f = st.radio(
                        f"Face-to-Face",
                        ['Dont Care', 'Yes', 'No'],
                        index=['Dont Care', 'Yes', 'No'].index(st.session_state.categorical_prefs[person_id]['F2F']),
                        key=get_key(person_id, 'F2F')
                    )
                    teams = st.radio(
                        f"Teams",
                        ['Dont Care', 'Yes', 'No'],
                        index=['Dont Care', 'Yes', 'No'].index(st.session_state.categorical_prefs[person_id]['Teams']),
                        key=get_key(person_id, 'Teams')
                    )
                    time_pref = st.selectbox(
                        f"Time of Day",
                        list(TIME_CHOICES.keys()),
                        index=list(TIME_CHOICES.keys()).index(st.session_state.categorical_prefs[person_id]['Time']),
                        key=get_key(person_id, 'Time')
                    )
                    st.session_state.categorical_prefs[person_id] = {
                        'Day_Preference': day_pref,
                        'F2F': f2f,
                        'Teams': teams,
                        'Time': time_pref,
                    }
        
        st.markdown("---")
        submitted = st.form_submit_button("Run Master Agent Negotiation")

    if submitted:
        st.sidebar.subheader("Input Review")
        if input_mode in ["Sliders (0-100)", "Both"]:
            st.sidebar.markdown("**Sliders**")
            slider_view = (
                pd.DataFrame(st.session_state.slider_prefs).T
                if app_mode == "Multi-user negotiation"
                else pd.DataFrame({target_person: st.session_state.slider_prefs[target_person]}).T
            )
            st.sidebar.dataframe(slider_view)
        if input_mode in ["Hardcoded (Yes/No)", "Both"]:
            st.sidebar.markdown("**Hardcoded**")
            cat_view = (
                pd.DataFrame(st.session_state.categorical_prefs).T
                if app_mode == "Multi-user negotiation"
                else pd.DataFrame({target_person: st.session_state.categorical_prefs[target_person]}).T
            )
            st.sidebar.dataframe(cat_view)

        results = []
        if app_mode == "Multi-user negotiation":
            if input_mode in ["Sliders (0-100)", "Both"]:
                final_assignments, log_messages = master_agent_negotiate(
                    st.session_state.slider_prefs,
                    scorer=calculate_utility_score_slider,
                    strength_fn=preference_strength_slider,
                    render_log=False
                )
                results.append(("Sliders (0-100)", final_assignments, log_messages, "slider", None))

            if input_mode in ["Hardcoded (Yes/No)", "Both"]:
                final_assignments, log_messages = master_agent_negotiate(
                    st.session_state.categorical_prefs,
                    scorer=calculate_utility_score_categorical,
                    strength_fn=preference_strength_categorical,
                    render_log=False
                )
                results.append(("Hardcoded (Yes/No)", final_assignments, log_messages, "categorical", None))
        else:
            if input_mode in ["Sliders (0-100)", "Both"]:
                other_assignments, occupied_slots, selected_others = simulate_other_bookings_random(
                    scorer=calculate_utility_score_slider,
                    strength_fn=preference_strength_slider,
                    target_person=target_person,
                    num_other_bookings=other_bookings,
                    mode_label="slider"
                )
                assigned_slot = assign_single_user(
                    st.session_state.slider_prefs,
                    target_person,
                    occupied_slots,
                    scorer=calculate_utility_score_slider
                )
                final_assignments = {target_person: assigned_slot} if assigned_slot else {}
                log_messages = [
                    f"Simulated other bookings: {', '.join(selected_others) if selected_others else 'none'} (random prefs).",
                    f"Occupied slots (before target): {', '.join(sorted(occupied_slots)) if occupied_slots else 'none'}.",
                ]
                results.append(("Sliders (0-100)", final_assignments, log_messages, "slider", occupied_slots))

            if input_mode in ["Hardcoded (Yes/No)", "Both"]:
                other_assignments, occupied_slots, selected_others = simulate_other_bookings_random(
                    scorer=calculate_utility_score_categorical,
                    strength_fn=preference_strength_categorical,
                    target_person=target_person,
                    num_other_bookings=other_bookings,
                    mode_label="categorical"
                )
                assigned_slot = assign_single_user(
                    st.session_state.categorical_prefs,
                    target_person,
                    occupied_slots,
                    scorer=calculate_utility_score_categorical
                )
                final_assignments = {target_person: assigned_slot} if assigned_slot else {}
                log_messages = [
                    f"Simulated other bookings: {', '.join(selected_others) if selected_others else 'none'} (random prefs).",
                    f"Occupied slots (before target): {', '.join(sorted(occupied_slots)) if occupied_slots else 'none'}.",
                ]
                results.append(("Hardcoded (Yes/No)", final_assignments, log_messages, "categorical", occupied_slots))
        
        st.header("Final Assignment Results")
        if len(results) > 1:
            tab_labels = [r[0] for r in results] + ["Inputs Review"]
            tabs = st.tabs(tab_labels)
        else:
            tabs = st.tabs(["Results", "Negotiation Log", "Inputs Review"])

        if len(results) > 1:
            for idx, (label, final_assignments, log_messages, mode, occupied_slots) in enumerate(results):
                with tabs[idx]:
                    if final_assignments:
                        if mode == "slider":
                            results_df = build_results_df(final_assignments, st.session_state.slider_prefs, mode)
                        else:
                            results_df = build_results_df(final_assignments, st.session_state.categorical_prefs, mode)

                        assigned_slot_ids = set(final_assignments.values())
                        blocked_slots = occupied_slots if occupied_slots else set()
                        unavailable = assigned_slot_ids.union(blocked_slots)
                        unassigned_slots = SLOTS_DF[~SLOTS_DF.index.isin(unavailable)].drop(columns=['Time'])

                        if app_mode == "Single-user booking (simulate others)":
                            st.success("Successfully assigned **1** user to a slot.")
                        else:
                            st.success(f"Successfully assigned **{len(results_df)}** out of 8 people to slots!")
                        render_report_section(results_df, len(unassigned_slots), f"({label})")
                        st.markdown("### Allocation Details")
                        st.dataframe(
                            results_df.style.format({
                                'Utility Score': '{:.2f}',
                                'Satisfaction (0-100)': '{:.1f}',
                            }),
                            use_container_width=True
                        )

                        if app_mode == "Single-user booking (simulate others)" and occupied_slots:
                            st.markdown("### Pre-booked Slots (Simulated Others)")
                            occupied_df = SLOTS_DF[SLOTS_DF.index.isin(occupied_slots)].drop(columns=['Time'])
                            st.dataframe(occupied_df, use_container_width=True)

                        st.markdown("### Unassigned Slots")
                        st.dataframe(unassigned_slots, use_container_width=True)
                    else:
                        st.error("The Master Agent failed to make any assignments.")

                    if log_messages:
                        st.markdown("### Negotiation Log")
                        st.code("\n".join(log_messages))
                    if app_mode == "Single-user booking (simulate others)":
                        st.info("Multi-agent simulation: other users book first, then the target user gets the best remaining slot.")
        else:
            results_tab, log_tab, inputs_tab = tabs
            label, final_assignments, log_messages, mode, occupied_slots = results[0]
            with results_tab:
                if final_assignments:
                    if mode == "slider":
                        results_df = build_results_df(final_assignments, st.session_state.slider_prefs, mode)
                    else:
                        results_df = build_results_df(final_assignments, st.session_state.categorical_prefs, mode)

                    assigned_slot_ids = set(final_assignments.values())
                    blocked_slots = occupied_slots if occupied_slots else set()
                    unavailable = assigned_slot_ids.union(blocked_slots)
                    unassigned_slots = SLOTS_DF[~SLOTS_DF.index.isin(unavailable)].drop(columns=['Time'])

                    if app_mode == "Single-user booking (simulate others)":
                        st.success("Successfully assigned **1** user to a slot.")
                    else:
                        st.success(f"Successfully assigned **{len(results_df)}** out of 8 people to slots!")
                    render_report_section(results_df, len(unassigned_slots), "")
                    st.markdown("### Allocation Details")
                    st.dataframe(
                        results_df.style.format({
                            'Utility Score': '{:.2f}',
                            'Satisfaction (0-100)': '{:.1f}',
                        }),
                        use_container_width=True
                    )

                    if app_mode == "Single-user booking (simulate others)" and occupied_slots:
                        st.markdown("### Pre-booked Slots (Simulated Others)")
                        occupied_df = SLOTS_DF[SLOTS_DF.index.isin(occupied_slots)].drop(columns=['Time'])
                        st.dataframe(occupied_df, use_container_width=True)

                    st.markdown("### Unassigned Slots")
                    st.dataframe(unassigned_slots, use_container_width=True)
                else:
                    st.error("The Master Agent failed to make any assignments.")

            with log_tab:
                if log_messages:
                    st.code("\n".join(log_messages))
                else:
                    st.info("No log messages to display.")
                if app_mode == "Single-user booking (simulate others)":
                    st.info("Multi-agent simulation: other users book first, then the target user gets the best remaining slot.")

            with inputs_tab:
                if mode == "slider":
                    st.dataframe(pd.DataFrame(st.session_state.slider_prefs).T)
                else:
                    st.dataframe(pd.DataFrame(st.session_state.categorical_prefs).T)

        if len(results) > 1:
            with tabs[-1]:
                if input_mode in ["Sliders (0-100)", "Both"]:
                    st.markdown("**Sliders**")
                    st.dataframe(pd.DataFrame(st.session_state.slider_prefs).T)
                if input_mode in ["Hardcoded (Yes/No)", "Both"]:
                    st.markdown("**Hardcoded**")
                    st.dataframe(pd.DataFrame(st.session_state.categorical_prefs).T)

    st.markdown("### Simulation: Sliders vs Hardcoded")
    st.caption("Runs 50 randomized preference sets using the same underlying preferences for both modes.")
    if st.button("Run 50x Simulation"):
        runs = 50
        threshold = 0.5 * SLIDER_MAX_SCORE
        sim_rows = []
        for run_id in range(1, runs + 1):
            slider_prefs = {}
            categorical_prefs = {}
            for p in PERSON_IDS:
                sp = {
                    'Pref_Mon': random.randint(0, 100),
                    'Pref_Tue': random.randint(0, 100),
                    'Pref_F2F': random.randint(0, 100),
                    'Pref_Teams': random.randint(0, 100),
                    'Pref_Time': random.randint(0, 100),
                }
                slider_prefs[p] = sp
                categorical_prefs[p] = slider_to_categorical(sp)

            slider_assignments, _ = master_agent_negotiate(
                slider_prefs,
                scorer=calculate_utility_score_slider,
                strength_fn=preference_strength_slider,
                render_log=False
            )
            cat_assignments, _ = master_agent_negotiate(
                categorical_prefs,
                scorer=calculate_utility_score_categorical,
                strength_fn=preference_strength_categorical,
                render_log=False
            )

            def eval_assignments(assignments):
                scores = []
                for person, slot_id in assignments.items():
                    scores.append(satisfaction_from_slider_prefs(slot_id, slider_prefs[person]))
                avg_score = float(np.mean(scores)) if scores else 0.0
                min_score = float(np.min(scores)) if scores else 0.0
                low_pct = float(np.mean([1 if s < threshold else 0 for s in scores])) if scores else 0.0
                return avg_score, low_pct, min_score

            s_avg, s_low, s_min = eval_assignments(slider_assignments)
            c_avg, c_low, c_min = eval_assignments(cat_assignments)
            sim_rows.append({
                'Run': run_id,
                'Mode': 'Sliders (0-100)',
                'Avg Satisfaction': normalize_slider_score(s_avg),
                '% Below Threshold': s_low * 100,
                'Min Satisfaction': normalize_slider_score(s_min),
            })
            sim_rows.append({
                'Run': run_id,
                'Mode': 'Hardcoded (Yes/No)',
                'Avg Satisfaction': normalize_slider_score(c_avg),
                '% Below Threshold': c_low * 100,
                'Min Satisfaction': normalize_slider_score(c_min),
            })

        sim_df = pd.DataFrame(sim_rows)
        sim_df = sim_df.round(1)
        summary = sim_df.groupby('Mode').agg({
            'Avg Satisfaction': ['mean', 'median', 'std'],
            '% Below Threshold': ['mean', 'median', 'std'],
            'Min Satisfaction': ['mean', 'median', 'std'],
        })
        summary.columns = [' '.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        summary = summary.round(1)

        st.subheader("Simulation Summary (50 runs)")
        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download Simulation Results (CSV)",
            sim_df.to_csv(index=False),
            file_name="simulation_results.csv",
            mime="text/csv",
        )

        st.markdown("### Distribution View")
        dist_cols = st.columns(3)
        with dist_cols[0]:
            st.area_chart(sim_df.pivot(index='Run', columns='Mode', values='Avg Satisfaction'))
            st.caption("Avg satisfaction per run")
        with dist_cols[1]:
            st.area_chart(sim_df.pivot(index='Run', columns='Mode', values='% Below Threshold'))
            st.caption("% below threshold per run")
        with dist_cols[2]:
            st.area_chart(sim_df.pivot(index='Run', columns='Mode', values='Min Satisfaction'))
            st.caption("Min satisfaction per run")

        st.markdown("### Distribution Tables (by method)")
        def dist_stats(series):
            return pd.Series({
                'min': series.min(),
                'p25': series.quantile(0.25),
                'median': series.median(),
                'p75': series.quantile(0.75),
                'max': series.max(),
            })

        dist_tables = sim_df.groupby('Mode').apply(
            lambda g: pd.concat([
                dist_stats(g['Avg Satisfaction']).add_prefix('Avg Satisfaction '),
                dist_stats(g['% Below Threshold']).add_prefix('% Below Threshold '),
                dist_stats(g['Min Satisfaction']).add_prefix('Min Satisfaction '),
            ])
        ).reset_index()
        dist_tables = dist_tables.round(1)
        method_cols = st.columns(2)
        slider_table = dist_tables[dist_tables['Mode'] == 'Sliders (0-100)']
        hard_table = dist_tables[dist_tables['Mode'] == 'Hardcoded (Yes/No)']
        with method_cols[0]:
            st.markdown("**Sliders (0-100)**")
            st.dataframe(slider_table, use_container_width=True)
        with method_cols[1]:
            st.markdown("**Hardcoded (Yes/No)**")
            st.dataframe(hard_table, use_container_width=True)

        st.markdown("### Best vs Worst Runs (Avg Satisfaction)")
        best_runs = sim_df.sort_values('Avg Satisfaction', ascending=False).groupby('Mode').head(3)
        worst_runs = sim_df.sort_values('Avg Satisfaction', ascending=True).groupby('Mode').head(3)
        bw_cols = st.columns(2)
        with bw_cols[0]:
            st.markdown("**Top 3 runs**")
            st.dataframe(best_runs.reset_index(drop=True), use_container_width=True)
        with bw_cols[1]:
            st.markdown("**Bottom 3 runs**")
            st.dataframe(worst_runs.reset_index(drop=True), use_container_width=True)

        if len(summary) == 2:
            slider_row = summary[summary['Mode'] == 'Sliders (0-100)'].iloc[0]
            hard_row = summary[summary['Mode'] == 'Hardcoded (Yes/No)'].iloc[0]
            winner_notes = []
            if slider_row['Avg Satisfaction mean'] > hard_row['Avg Satisfaction mean']:
                winner_notes.append("higher average satisfaction")
            if slider_row['% Below Threshold mean'] < hard_row['% Below Threshold mean']:
                winner_notes.append("fewer low-satisfaction assignments")
            if slider_row['Min Satisfaction mean'] > hard_row['Min Satisfaction mean']:
                winner_notes.append("better worst-case satisfaction")

            if winner_notes:
                st.success("Sliders win on: " + ", ".join(winner_notes) + ".")
            else:
                st.info("No clear winner across the three metrics.")


if __name__ == "__main__":
    main()
