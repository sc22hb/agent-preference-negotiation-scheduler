const DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"];
const PERIOD_ORDER = ["morning", "afternoon", "evening"];
const MODALITY_ORDER = ["in_person", "phone", "video"];

const state = {
  runId: null,
  viewMode: "user",
  captureMode: "conversation",
  humanConversationMode: false,
  conversationBaseText: null,
  clarificationAnswers: {},
  intakeSummary: null,
  routingDecision: null,
  preferences: null,
  admin: {
    intake: null,
    route: null,
    scheduling: [],
    relaxTrail: [],
    latestSlotScores: [],
    rotaTimetable: null,
  },
};

const thread = document.getElementById("chatThread");
const stateSummary = document.getElementById("stateSummary");
const adminIntake = document.getElementById("adminIntake");
const adminRoute = document.getElementById("adminRoute");
const adminScheduling = document.getElementById("adminScheduling");
const adminRelaxTrail = document.getElementById("adminRelaxTrail");
const adminSlotScores = document.getElementById("adminSlotScores");
const adminRotaTimetable = document.getElementById("adminRotaTimetable");
const auditPageLink = document.getElementById("auditPageLink");
const auditJsonLink = document.getElementById("auditJsonLink");
const runIdView = document.getElementById("runIdView");
const adminRunIdView = document.getElementById("adminRunIdView");

const pageEyebrow = document.getElementById("pageEyebrow");
const pageTitle = document.getElementById("pageTitle");
const pageSubtitle = document.getElementById("pageSubtitle");

const userViewToggle = document.getElementById("userViewToggle");
const adminViewToggle = document.getElementById("adminViewToggle");
const resetButton = document.getElementById("btnReset");

const captureModeSelect = document.getElementById("captureMode");
const captureModeConversationButton = document.getElementById("captureModeConversation");
const captureModeFormButton = document.getElementById("captureModeForm");
const humanModeToggle = document.getElementById("humanModeToggle");
const humanModeToggleLabel = document.getElementById("humanModeToggleLabel");
const conversationInputs = document.getElementById("conversationInputs");
const formInputs = document.getElementById("formInputs");
const captureModeLabel = document.getElementById("captureModeLabel");
const userTextLabel = document.getElementById("userTextLabel");
const conversationHint = document.getElementById("conversationHint");
const formReasonLabel = document.getElementById("formReasonLabel");
const formPreferredModalityLabel = document.getElementById("formPreferredModalityLabel");
const modalityLegend = document.getElementById("modalityLegend");
const availabilityLegend = document.getElementById("availabilityLegend");
const formHorizonLabel = document.getElementById("formHorizonLabel");
const formHorizonHint = document.getElementById("formHorizonHint");
const formSoonestWeightLabel = document.getElementById("formSoonestWeightLabel");
const formSoonestWeightHint = document.getElementById("formSoonestWeightHint");
const formGlobalHint = document.getElementById("formGlobalHint");

const userStatusHeading = document.getElementById("userStatusHeading");
const userStatusBody = document.getElementById("userStatusBody");
const userSummaryContent = document.getElementById("userSummaryContent");
const userReasonSummary = document.getElementById("userReasonSummary");
const userServiceSummary = document.getElementById("userServiceSummary");
const userSearchSummary = document.getElementById("userSearchSummary");
const userPreferenceChips = document.getElementById("userPreferenceChips");
const userBookingState = document.getElementById("userBookingState");

const formHorizonInput = document.getElementById("formHorizon");
const formSoonestWeightInput = document.getElementById("formSoonestWeight");
const formHorizonValue = document.getElementById("formHorizonValue");
const formSoonestWeightValue = document.getElementById("formSoonestWeightValue");
const formHorizonThumbValue = document.getElementById("formHorizonThumbValue");
const formSoonestWeightThumbValue = document.getElementById("formSoonestWeightThumbValue");

const intakeButton = document.getElementById("btnIntake");

function formatDayCount(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return "10 days";
  }
  return `${parsed} day${parsed === 1 ? "" : "s"}`;
}

function titleCase(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function prettifyModality(modality) {
  return titleCase(modality);
}

function prettifyService(service) {
  return titleCase(service);
}

function joinList(values) {
  const items = (values || []).filter(Boolean);
  if (items.length === 0) {
    return "";
  }
  if (items.length === 1) {
    return items[0];
  }
  if (items.length === 2) {
    return `${items[0]} and ${items[1]}`;
  }
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function formatDateTime(dateValue, options = {}) {
  const date = new Date(dateValue);
  return date.toLocaleString(undefined, options);
}

function formatDateOnly(dateValue) {
  const date = new Date(`${dateValue}T12:00:00`);
  return date.toLocaleDateString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function positionSliderThumbValue(input, thumb) {
  if (!input || !thumb) {
    return;
  }
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value || min);
  const percent = max === min ? 0 : ((value - min) / (max - min)) * 100;
  thumb.style.left = `calc(${percent}% + (${10 - percent * 0.2}px))`;
}

function updateFormSliderLabels() {
  if (formHorizonValue && formHorizonInput) {
    formHorizonValue.textContent = formatDayCount(formHorizonInput.value);
  }
  if (formSoonestWeightValue && formSoonestWeightInput) {
    formSoonestWeightValue.textContent = String(formSoonestWeightInput.value);
  }
  if (formHorizonThumbValue && formHorizonInput) {
    formHorizonThumbValue.textContent = String(formHorizonInput.value);
    positionSliderThumbValue(formHorizonInput, formHorizonThumbValue);
  }
  if (formSoonestWeightThumbValue && formSoonestWeightInput) {
    formSoonestWeightThumbValue.textContent = String(formSoonestWeightInput.value);
    positionSliderThumbValue(formSoonestWeightInput, formSoonestWeightThumbValue);
  }
}

function isFriendlyMode() {
  return state.viewMode === "user" || Boolean(state.humanConversationMode);
}

function getLatestSchedulingResponse() {
  if (!state.admin.scheduling.length) {
    return null;
  }
  return state.admin.scheduling[state.admin.scheduling.length - 1];
}

function updateRunIdDisplay() {
  const runText = `Run ID: ${state.runId || "not started"}`;
  runIdView.textContent = runText;
  if (adminRunIdView) {
    adminRunIdView.textContent = runText;
  }

  const hasRun = Boolean(state.runId);
  if (hasRun) {
    auditPageLink.href = `/audit/${state.runId}`;
    auditJsonLink.href = `/api/run/${state.runId}/audit`;
    auditJsonLink.download = `audit-${state.runId}.json`;
    auditPageLink.classList.remove("disabled");
    auditJsonLink.classList.remove("disabled");
    return;
  }

  auditPageLink.href = "#";
  auditJsonLink.href = "#";
  auditJsonLink.removeAttribute("download");
  auditPageLink.classList.add("disabled");
  auditJsonLink.classList.add("disabled");
}

function appendAdminDebugMessage(title, content) {
  const bubble = document.createElement("div");
  bubble.className = "bubble system admin-debug-bubble";
  bubble.dataset.adminDebug = "true";

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  titleEl.textContent = title;

  const bodyEl = document.createElement("div");
  bodyEl.textContent = content;

  bubble.appendChild(titleEl);
  bubble.appendChild(bodyEl);
  thread.appendChild(bubble);
}

function clearAdminDebugMessages() {
  thread.querySelectorAll('[data-admin-debug="true"]').forEach((node) => node.remove());
}

function renderAdminTranscript() {
  clearAdminDebugMessages();

  if (state.viewMode !== "admin") {
    return;
  }

  appendAdminDebugMessage(
    "Admin Trace",
    state.runId
      ? "Technical trace for the current run. The user-facing bubbles stay hidden in admin mode."
      : "Start a request in User View to generate an audit trail."
  );

  if (!state.runId) {
    return;
  }

  if (state.admin.intake) {
    appendAdminDebugMessage("Intake Response", JSON.stringify(state.admin.intake, null, 2));

    if (state.admin.intake.extraction_engine === "llm" && state.admin.intake.llm_response) {
      appendAdminDebugMessage("LLM Parsed JSON", JSON.stringify(state.admin.intake.llm_response, null, 2));
    }
  }

  if (state.intakeSummary) {
    appendAdminDebugMessage("Structured Intake", JSON.stringify(state.intakeSummary, null, 2));
  }

  if (state.admin.route) {
    appendAdminDebugMessage("Routing Decision", JSON.stringify(state.admin.route, null, 2));
  }

  if (state.admin.scheduling.length) {
    state.admin.scheduling.forEach((round, index) => {
      const roundLabel = round.preview
        ? `Scheduling Preview ${index + 1}`
        : `Scheduling Round ${round.round_number || index + 1}`;

      appendAdminDebugMessage(roundLabel, JSON.stringify(round, null, 2));

      if (round.slot_scores?.length) {
        appendAdminDebugMessage(`${roundLabel} Slot Scores`, formatSlotScores(round.slot_scores));
      }

      if (round.booking) {
        appendAdminDebugMessage(`${roundLabel} Booking`, JSON.stringify(round.booking, null, 2));
      }

      if (round.blocker_summary) {
        appendAdminDebugMessage(
          `${roundLabel} Blockers`,
          JSON.stringify(round.blocker_summary, null, 2)
        );
      }

      if (round.relaxation_questions?.length) {
        appendAdminDebugMessage(
          `${roundLabel} Relaxation Questions`,
          JSON.stringify(round.relaxation_questions, null, 2)
        );
      }
    });
  }

  if (state.admin.relaxTrail.length) {
    state.admin.relaxTrail.forEach((relaxation, index) => {
      appendAdminDebugMessage(
        `Relaxation Round ${index + 1}`,
        JSON.stringify(relaxation, null, 2)
      );
    });
  }
}

function updateViewCopy() {
  const isAdmin = state.viewMode === "admin";
  const userTextInput = document.getElementById("userText");
  const intakeButtonLabel = isAdmin ? "Submit Intake" : "Find Appointment";

  pageEyebrow.hidden = isAdmin;
  pageTitle.textContent = isAdmin
    ? "NHS-Style Multi-Agent Scheduling Demo"
    : "Book an appointment";
  pageSubtitle.textContent = isAdmin
    ? "Synthetic data only. Technical demonstrator."
    : "Send a short message or use the guided form. We’ll help find the best available slot.";

  captureModeLabel.textContent = "Interface";
  humanModeToggleLabel.textContent = isAdmin
    ? "Human conversation mode (hide parsing details)"
    : "Human conversation mode (hide parsing details)";
  userTextLabel.textContent = isAdmin ? "Your message" : "What do you need help with?";
  userTextInput.placeholder = isAdmin
    ? "Example: persistent cough for 2 weeks, no Monday mornings, phone preferred"
    : "Example: I’ve had a persistent cough for 2 weeks, phone is easiest, and Monday mornings do not work for me.";
  conversationHint.textContent = isAdmin
    ? ""
    : "Include symptoms, how long it has been going on, and any timing preferences that matter.";
  conversationHint.hidden = isAdmin;

  formReasonLabel.textContent = isAdmin ? "Reason" : "What do you need?";
  formPreferredModalityLabel.textContent = isAdmin ? "Preferred modality" : "Preferred appointment type";
  modalityLegend.textContent = isAdmin
    ? "Hard constraints: modalities you cannot do"
    : "Appointment types that will not work";
  availabilityLegend.textContent = isAdmin
    ? "Day + time preferences and hard constraints"
    : "Choose the days and times you prefer, and anything that must be avoided";

  formHorizonLabel.childNodes[0].textContent = isAdmin
    ? "How many days ahead should we search? "
    : "How far ahead should we look? ";
  formHorizonHint.textContent = isAdmin
    ? "Lower means a tighter search window. Higher checks further ahead."
    : "A wider search window checks more appointments.";
  formSoonestWeightLabel.childNodes[0].textContent = isAdmin
    ? "How important is getting the soonest appointment? "
    : "How important is the earliest slot? ";
  formSoonestWeightHint.textContent = isAdmin
    ? "0 ignores soonness. 100 strongly prioritises earliest slots."
    : "Higher values push the search toward the soonest feasible slot.";
  formGlobalHint.textContent = isAdmin
    ? "Hard blocks always win. Day+time preferences feed slot scoring in a more precise way than global day/period picks."
    : "Hard blocks always win over preferences.";

  intakeButton.textContent = intakeButtonLabel;
}

function setRunId(runId) {
  state.runId = runId;
  updateRunIdDisplay();
}

function addMessage(role, title, content) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  if (isFriendlyMode() && role === "user") {
    titleEl.textContent = "You";
  } else if (isFriendlyMode() && role === "assistant") {
    titleEl.textContent = "Scheduler Assistant";
  } else {
    titleEl.textContent = title;
  }

  const bodyEl = document.createElement("div");
  bodyEl.textContent = content;

  bubble.appendChild(titleEl);
  bubble.appendChild(bodyEl);
  thread.appendChild(bubble);
  thread.scrollTop = thread.scrollHeight;
}

function appendBookingNote(parent, title, message, tone = "default") {
  const wrap = document.createElement("div");
  wrap.className = "booking-note-wrap";
  if (tone === "warning") {
    wrap.classList.add("warning");
  }
  if (tone === "error") {
    wrap.classList.add("error");
  }

  const titleEl = document.createElement("h3");
  titleEl.className = "booking-note-title";
  titleEl.textContent = title;

  const messageEl = document.createElement("p");
  messageEl.className = "booking-note";
  messageEl.textContent = message;

  wrap.appendChild(titleEl);
  wrap.appendChild(messageEl);
  parent.appendChild(wrap);
}

function renderChipList(container, chips) {
  container.innerHTML = "";
  chips.forEach((chipText) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = chipText;
    container.appendChild(chip);
  });
}

function buildPreferenceChips(preferences) {
  if (!preferences) {
    return [];
  }

  const chips = [];
  const compactMode = isFriendlyMode();

  if (preferences.preferred_modalities?.length) {
    chips.push(`Prefers ${joinList(preferences.preferred_modalities.map(prettifyModality))}`);
  }
  if (preferences.excluded_modalities?.length) {
    chips.push(`Avoids ${joinList(preferences.excluded_modalities.map(prettifyModality))}`);
  }
  if (preferences.preferred_days?.length) {
    chips.push(`Best days: ${preferences.preferred_days.join(", ")}`);
  }
  if (preferences.excluded_days?.length) {
    chips.push(`Unavailable: ${preferences.excluded_days.join(", ")}`);
  }
  if (preferences.preferred_day_periods?.length) {
    chips.push(`${preferences.preferred_day_periods.length} preferred time windows`);
  }
  if (preferences.excluded_day_periods?.length) {
    chips.push(`${preferences.excluded_day_periods.length} blocked time windows`);
  }
  if (preferences.earliest_start_date) {
    chips.push(
      compactMode
        ? `Starts from ${formatDateOnly(preferences.earliest_start_date)}`
        : `Earliest date: ${formatDateOnly(preferences.earliest_start_date)}`
    );
  }

  if (compactMode) {
    chips.push(`Looking ahead ${formatDayCount(preferences.date_horizon_days)}`);
  } else {
    chips.push(`Search window: ${formatDayCount(preferences.date_horizon_days)}`);
    chips.push(`Soonest priority: ${preferences.soonest_weight}/100`);
  }
  return chips;
}

function summarizeSearch(preferences) {
  if (!preferences) {
    return "No preferences captured yet";
  }
  if (preferences.earliest_start_date) {
    return isFriendlyMode()
      ? `We are checking availability from ${formatDateOnly(preferences.earliest_start_date)} onward.`
      : `Searching from ${formatDateOnly(preferences.earliest_start_date)} onward with earliest-slot priority at ${preferences.soonest_weight}/100`;
  }
  if (isFriendlyMode()) {
    return `We are checking availability across the next ${formatDayCount(preferences.date_horizon_days)}.`;
  }
  return `Searching ${formatDayCount(preferences.date_horizon_days)} ahead with earliest-slot priority at ${preferences.soonest_weight}/100`;
}

function updateUserExperience() {
  const intakeResponse = state.admin.intake;
  const intakeSummary = state.intakeSummary;
  const latestScheduling = getLatestSchedulingResponse();

  let heading = "Start your request";
  let body =
    "Tell us what you need in plain language, or switch to the guided form if you prefer step-by-step help.";

  if (intakeResponse?.safety?.triggered) {
    heading = "Urgent follow-up needed";
    body = intakeResponse.safety.message || "This request needs human review before any booking should continue.";
  } else if (latestScheduling?.status === "booked" && latestScheduling.booking) {
    heading = "Appointment ready";
    body = "We have found the best available slot based on your request and current availability.";
  } else if (latestScheduling?.status === "needs_relaxation") {
    heading = "One more choice needed";
    body =
      latestScheduling.message ||
      "No exact match was available. If you can relax one constraint, the search can continue.";
  } else if (latestScheduling?.status === "failed") {
    heading = "No suitable appointment found";
    body = latestScheduling.message || "The current search window and constraints did not produce a feasible appointment.";
  } else if (state.routingDecision || intakeSummary) {
    heading = "Checking availability";
    body = "Your request has been captured and we are checking the best matching appointments.";
  }

  userStatusHeading.textContent = heading;
  userStatusBody.textContent = body;

  if (intakeSummary) {
    userSummaryContent.hidden = false;
    userReasonSummary.textContent = titleCase(intakeSummary.complaint_category);
    userServiceSummary.textContent = state.routingDecision
      ? isFriendlyMode()
        ? prettifyService(state.routingDecision.service_type)
        : `${prettifyService(state.routingDecision.service_type)} | ${titleCase(state.routingDecision.urgency_band)} priority`
      : "Waiting for triage";
    userSearchSummary.textContent = summarizeSearch(state.preferences || intakeSummary.preferences);
    renderChipList(userPreferenceChips, buildPreferenceChips(state.preferences || intakeSummary.preferences));
  } else {
    userSummaryContent.hidden = true;
    userPreferenceChips.innerHTML = "";
  }

  userBookingState.innerHTML = "";
  userBookingState.classList.remove("empty-state");
  userBookingState.hidden = false;

  if (!state.runId) {
    userBookingState.hidden = true;
    return;
  }

  if (intakeResponse?.safety?.triggered) {
    appendBookingNote(
      userBookingState,
      "Booking paused",
      intakeResponse.safety.message || "This request should be reviewed urgently by a human clinician or receptionist.",
      "error"
    );
    return;
  }

  if (latestScheduling?.status === "booked" && latestScheduling.booking) {
    const highlight = document.createElement("div");
    highlight.className = "booking-highlight";

    const title = document.createElement("h3");
    title.textContent = formatDateTime(latestScheduling.booking.start_time, {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });

    const subtitle = document.createElement("p");
    subtitle.textContent = isFriendlyMode()
      ? `${prettifyModality(latestScheduling.booking.modality)} appointment`
      : `${prettifyModality(latestScheduling.booking.modality)} ${prettifyService(latestScheduling.booking.service_type)} appointment`;

    highlight.appendChild(title);
    highlight.appendChild(subtitle);
    userBookingState.appendChild(highlight);

    const meta = document.createElement("div");
    meta.className = "booking-meta";

    const site = document.createElement("strong");
    site.textContent = latestScheduling.booking.site;

    meta.appendChild(site);
    if (!isFriendlyMode()) {
      const clinician = document.createElement("span");
      clinician.textContent = `Clinician: ${latestScheduling.booking.clinician_id}`;
      const utility = document.createElement("span");
      utility.textContent = `Match score: ${latestScheduling.booking.utility}`;
      meta.appendChild(clinician);
      meta.appendChild(utility);
    }
    userBookingState.appendChild(meta);

    appendBookingNote(
      userBookingState,
      isFriendlyMode() ? "What happens next" : "What happens next",
      isFriendlyMode()
        ? "If you need to change anything, start a new request."
        : "Use Admin View if you want the full audit, route, and candidate slot scoring behind this result."
    );
    return;
  }

  if (latestScheduling?.status === "needs_relaxation") {
    appendBookingNote(
      userBookingState,
      "More flexibility may help",
      latestScheduling.message || "No exact match was found. Review the latest assistant prompt and relax a constraint if you want another attempt.",
      "warning"
    );
    return;
  }

  if (latestScheduling?.status === "failed") {
    appendBookingNote(
      userBookingState,
      "No appointment found",
      latestScheduling.message || "The current search did not produce a suitable appointment.",
      "error"
    );
    return;
  }

  appendBookingNote(
    userBookingState,
    "Checking appointments",
    "The request has been captured and the search is in progress."
  );
}

function updateViewMode() {
  document.body.classList.toggle("view-user", state.viewMode === "user");
  document.body.classList.toggle("view-admin", state.viewMode === "admin");
  userViewToggle.classList.toggle("active", state.viewMode === "user");
  adminViewToggle.classList.toggle("active", state.viewMode === "admin");
  updateViewCopy();
  updatePresentationMode();
  renderAdminTranscript();
}

function setViewMode(mode) {
  state.viewMode = mode === "admin" ? "admin" : "user";
  updateViewMode();
}

function updateCaptureButtons() {
  captureModeConversationButton.classList.toggle("active", state.captureMode === "conversation");
  captureModeFormButton.classList.toggle("active", state.captureMode === "form");
  captureModeSelect.value = state.captureMode;
}

function setCaptureMode(mode, options = {}) {
  const { resetConversationBase = true } = options;
  const nextMode = mode === "form" ? "form" : "conversation";
  if (resetConversationBase && state.captureMode !== nextMode) {
    state.conversationBaseText = null;
  }

  state.captureMode = nextMode;
  conversationInputs.style.display = state.captureMode === "conversation" ? "block" : "none";
  formInputs.style.display = state.captureMode === "form" ? "block" : "none";
  updateCaptureButtons();
  updateAdminView();
}

function updatePresentationMode() {
  state.humanConversationMode = Boolean(humanModeToggle?.checked);
  document.body.classList.toggle("friendly-flow", isFriendlyMode());
  updateUserExperience();
}

function addClarificationFormBubble(questions) {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  titleEl.textContent = isFriendlyMode() ? "Scheduler Assistant" : "Assistant";

  const textEl = document.createElement("div");
  textEl.textContent = isFriendlyMode()
    ? "I need a little more detail before I look for an appointment."
    : "I need a couple of clarifications before booking:";

  const formEl = document.createElement("div");
  formEl.className = "clarify-form";

  questions.forEach((question, index) => {
    const label = document.createElement("label");
    label.className = "clarify-label";
    label.textContent = `${index + 1}. ${question.prompt}`;

    const input = document.createElement("input");
    input.className = "clarify-input";
    input.setAttribute("data-clarify-id", question.question_id);
    input.placeholder = "Type your answer";

    formEl.appendChild(label);
    formEl.appendChild(input);
  });

  const sendButton = document.createElement("button");
  sendButton.type = "button";
  sendButton.textContent = isFriendlyMode() ? "Continue" : "Send Answers";

  sendButton.addEventListener("click", async () => {
    const answers = {};
    let missingAnswer = false;

    formEl.querySelectorAll("[data-clarify-id]").forEach((input) => {
      const value = input.value.trim();
      const questionId = input.getAttribute("data-clarify-id");
      if (!value) {
        missingAnswer = true;
      } else if (questionId) {
        answers[questionId] = value;
      }
    });

    if (missingAnswer) {
      addMessage("system", "System", "Please answer all clarification questions before continuing.");
      return;
    }

    sendButton.disabled = true;
    if (isFriendlyMode()) {
      addMessage("user", "Patient", Object.values(answers).join("\n"));
    } else {
      addMessage("user", "Patient", `Clarifications: ${JSON.stringify(answers)}`);
    }

    try {
      const response = await submitConversationIntake(answers, true);
      await handleIntakeResponse(response);
      sendButton.textContent = isFriendlyMode() ? "Sent" : "Answers Sent";
    } catch (error) {
      sendButton.disabled = false;
      addMessage("system", "System", error.message);
    }
  });

  formEl.appendChild(sendButton);
  bubble.appendChild(titleEl);
  bubble.appendChild(textEl);
  bubble.appendChild(formEl);
  thread.appendChild(bubble);
  thread.scrollTop = thread.scrollHeight;
}

function addRelaxationFormBubble(questions, blockerSummary, currentPreferences) {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  titleEl.textContent = isFriendlyMode() ? "Scheduler Assistant" : "Assistant";

  const textEl = document.createElement("div");
  textEl.textContent = isFriendlyMode()
    ? "I could not find an exact slot yet. Tell me what can flex and I will try again."
    : "No exact slot yet. Choose which constraints to relax for the next round:";

  const formEl = document.createElement("div");
  formEl.className = "clarify-form";
  const allDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  questions.forEach((question, index) => {
    const label = document.createElement("label");
    label.className = "clarify-label";
    label.textContent = `${index + 1}. ${question.prompt}`;

    const select = document.createElement("select");
    select.className = "clarify-input";
    select.setAttribute("data-relax-id", question.key);

    const no = document.createElement("option");
    no.value = "false";
    no.textContent = "No";

    const yes = document.createElement("option");
    yes.value = "true";
    yes.textContent = "Yes";

    select.appendChild(no);
    select.appendChild(yes);

    formEl.appendChild(label);
    formEl.appendChild(select);

    if (question.key === "relax_excluded_days") {
      const excludedDays = Array.isArray(currentPreferences?.excluded_days) ? currentPreferences.excluded_days : [];
      const excludedDayPeriods = Array.isArray(currentPreferences?.excluded_day_periods)
        ? currentPreferences.excluded_day_periods
        : [];
      const pairedDays = excludedDayPeriods
        .map((pair) => pair?.day)
        .filter((day) => typeof day === "string");
      const candidateDays = allDays.filter((day) => new Set([...excludedDays, ...pairedDays]).has(day));

      if (candidateDays.length) {
        const daySelectLabel = document.createElement("label");
        daySelectLabel.className = "clarify-label";
        daySelectLabel.textContent = "Pick days to relax:";
        formEl.appendChild(daySelectLabel);

        const daySelectWrap = document.createElement("div");
        daySelectWrap.className = "checkbox-grid";

        candidateDays.forEach((day) => {
          const option = document.createElement("label");
          option.className = "checkbox-item";

          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.setAttribute("data-relax-day-key", question.key);
          checkbox.value = day;

          option.appendChild(checkbox);
          option.appendChild(document.createTextNode(day));
          daySelectWrap.appendChild(option);
        });

        formEl.appendChild(daySelectWrap);
      }
    }
  });

  if (!isFriendlyMode() && blockerSummary?.ranked_reasons?.length) {
    const blockersEl = document.createElement("div");
    blockersEl.className = "clarify-label";
    blockersEl.textContent = `Main blockers: ${blockerSummary.ranked_reasons.join(", ")}`;
    formEl.appendChild(blockersEl);
  }

  const sendButton = document.createElement("button");
  sendButton.type = "button";
  sendButton.textContent = isFriendlyMode() ? "Try Again" : "Apply and Retry";

  sendButton.addEventListener("click", async () => {
    const answers = {};
    const relaxationSelections = {};

    formEl.querySelectorAll("[data-relax-id]").forEach((input) => {
      const key = input.getAttribute("data-relax-id");
      answers[key] = input.value === "true";
    });

    if (answers.relax_excluded_days) {
      const selectedDays = Array.from(
        formEl.querySelectorAll('[data-relax-day-key="relax_excluded_days"]:checked')
      ).map((input) => input.value);
      const hasDayOptions = formEl.querySelectorAll('[data-relax-day-key="relax_excluded_days"]').length > 0;

      if (hasDayOptions && selectedDays.length === 0) {
        addMessage("system", "System", "Choose at least one day to relax, or set day relaxation to No.");
        return;
      }

      if (selectedDays.length > 0) {
        relaxationSelections.relax_excluded_days = selectedDays;
      }
    }

    sendButton.disabled = true;
    if (isFriendlyMode()) {
      const approved = Object.entries(answers)
        .filter((entry) => entry[1])
        .map((entry) => entry[0]);
      const message = approved.length
        ? `Please relax: ${approved.join(", ")}`
        : "Keep the current preferences.";
      addMessage("user", "Patient", message);
    } else {
      addMessage("user", "Patient", `Relaxation choices: ${JSON.stringify(answers)}`);
    }

    try {
      const relaxResponse = await postJson("/api/schedule/relax", {
        run_id: state.runId,
        preferences: currentPreferences,
        answers,
        relaxation_selections: relaxationSelections,
      });

      state.admin.relaxTrail.push(relaxResponse);
      state.preferences = relaxResponse.updated_preferences;
      updateAdminView();
      updateUserExperience();

      await runSchedulingRound(state.preferences);
      sendButton.textContent = isFriendlyMode() ? "Applied" : "Applied";
    } catch (error) {
      sendButton.disabled = false;
      addMessage("system", "System", error.message);
    }
  });

  formEl.appendChild(sendButton);
  bubble.appendChild(titleEl);
  bubble.appendChild(textEl);
  bubble.appendChild(formEl);
  thread.appendChild(bubble);
  thread.scrollTop = thread.scrollHeight;
}

function updateAdminView() {
  stateSummary.textContent = JSON.stringify(
    {
      run_id: state.runId,
      view_mode: state.viewMode,
      capture_mode: state.captureMode,
      has_intake: Boolean(state.intakeSummary),
      has_route: Boolean(state.routingDecision),
      has_preferences: Boolean(state.preferences),
    },
    null,
    2
  );

  adminIntake.textContent = state.admin.intake
    ? JSON.stringify(state.admin.intake, null, 2)
    : "No intake response yet.";
  adminRoute.textContent = state.admin.route
    ? JSON.stringify(state.admin.route, null, 2)
    : "No route response yet.";
  adminScheduling.textContent = state.admin.scheduling.length
    ? JSON.stringify(state.admin.scheduling, null, 2)
    : "No scheduling responses yet.";
  adminRotaTimetable.textContent = state.admin.rotaTimetable
    ? formatRotaTimetable(state.admin.rotaTimetable)
    : "Loading timetable...";
  adminSlotScores.textContent = state.admin.latestSlotScores.length
    ? JSON.stringify(state.admin.latestSlotScores, null, 2)
    : "No slot scoring yet.";
  adminRelaxTrail.textContent = state.admin.relaxTrail.length
    ? JSON.stringify(state.admin.relaxTrail, null, 2)
    : "No relaxations yet.";

  renderAdminTranscript();
}

function formatRotaTimetable(payload) {
  if (!payload || !payload.services) {
    return "No timetable available.";
  }

  const lines = [];
  lines.push(`Hospital: ${payload.hospital}`);
  lines.push(`DB Horizon: ${payload.database_horizon_days} days`);
  lines.push(`Database Builds: ${payload.database_build_count}`);

  const serviceKeys = Object.keys(payload.services).sort();
  serviceKeys.forEach((service) => {
    const slots = payload.services[service] || [];
    lines.push("");
    lines.push(`${service} (${slots.length} slots)`);
    slots.forEach((slot, index) => {
      const dt = new Date(slot.start_time);
      const weekday = dt.toLocaleDateString(undefined, { weekday: "short" });
      const when = `${weekday} ${dt.toLocaleDateString()} ${dt.toLocaleTimeString()}`;
      lines.push(`${index + 1}. ${when} | ${slot.modality} | ${slot.clinician_id} | ${slot.slot_id}`);
    });
  });

  return lines.join("\n");
}

function resetAdminRoundData() {
  state.admin.route = null;
  state.admin.scheduling = [];
  state.admin.relaxTrail = [];
  state.admin.latestSlotScores = [];
  state.routingDecision = null;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || response.statusText || "Request failed");
  }

  return response.json();
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || response.statusText || "Request failed");
  }
  return response.json();
}

function buildExtractorFields(mode) {
  if (mode === "form") {
    return {
      extractor: "rule",
      api_key: null,
      llm_model: null,
    };
  }

  return {
    extractor: "llm",
    api_key: null,
    llm_model: null,
  };
}

function buildFormPreferences() {
  const preferredModality = document.getElementById("formPreferredModality").value;
  const excludedModalities = MODALITY_ORDER.filter((modality) =>
    Boolean(document.getElementById(`hard-modality-${modality}`)?.checked)
  );
  const preferredModalities = preferredModality && !excludedModalities.includes(preferredModality)
    ? [preferredModality]
    : [];
  const dateHorizon = Number(formHorizonInput?.value || 10);
  const soonestWeight = Number(formSoonestWeightInput?.value || 100);
  const preferredDays = [];
  const excludedDays = [];
  const preferredDayPeriods = [];
  const excludedDayPeriods = [];

  DAY_ORDER.forEach((day) => {
    const preferredDayChecked = Boolean(document.getElementById(`pref-day-${day}`)?.checked);
    const hardDayChecked = Boolean(document.getElementById(`hard-day-${day}`)?.checked);

    if (hardDayChecked) {
      excludedDays.push(day);
    } else if (preferredDayChecked) {
      preferredDays.push(day);
    }

    PERIOD_ORDER.forEach((period) => {
      const preferredPeriodChecked = Boolean(document.getElementById(`pref-slot-${day}-${period}`)?.checked);
      const hardPeriodChecked = Boolean(document.getElementById(`hard-slot-${day}-${period}`)?.checked);

      if (hardDayChecked || hardPeriodChecked) {
        if (!hardDayChecked && hardPeriodChecked) {
          excludedDayPeriods.push({ day, period });
        }
        return;
      }

      if (preferredPeriodChecked) {
        preferredDayPeriods.push({ day, period });
      }
    });
  });

  return {
    preferred_modalities: preferredModalities,
    excluded_modalities: excludedModalities,
    preferred_days: preferredDays,
    excluded_days: excludedDays,
    preferred_periods: [],
    excluded_periods: [],
    preferred_day_periods: preferredDayPeriods,
    excluded_day_periods: excludedDayPeriods,
    date_horizon_days: Math.max(1, Math.min(30, dateHorizon)),
    soonest_weight: Math.max(0, Math.min(100, soonestWeight)),
    flexibility: {
      allow_time_relax: true,
      allow_modality_relax: true,
      allow_date_horizon_relax: true,
    },
  };
}

async function submitConversationIntake(clarificationAnswers, isFollowUp = false) {
  const userTextInput = document.getElementById("userText");
  const enteredText = userTextInput.value.trim();

  if (!state.conversationBaseText) {
    if (!enteredText) {
      throw new Error("Please enter a message first.");
    }
    state.conversationBaseText = enteredText;
    state.clarificationAnswers = {};
  }

  if (!isFollowUp) {
    addMessage("user", "Patient", state.conversationBaseText);
  }

  if (isFollowUp) {
    if (!state.runId) {
      throw new Error("Missing run id for intake refinement.");
    }
    const mergedAnswers = {
      ...state.clarificationAnswers,
      ...clarificationAnswers,
    };
    state.clarificationAnswers = mergedAnswers;
    return postJson("/api/intake/refine", {
      run_id: state.runId,
      clarification_answers: mergedAnswers,
      ...buildExtractorFields("conversation"),
    });
  }

  state.clarificationAnswers = { ...clarificationAnswers };
  return postJson("/api/intake", {
    user_text: state.conversationBaseText,
    clarification_answers: state.clarificationAnswers,
    ...buildExtractorFields("conversation"),
  });
}

async function submitFormIntake() {
  const reasonCode = document.getElementById("formReasonCode").value;
  if (!reasonCode) {
    throw new Error("Please choose a reason first.");
  }

  addMessage(
    "user",
    "Patient",
    isFriendlyMode() ? "I submitted the guided form." : "Submitted form preferences."
  );

  return postJson("/api/intake/form", {
    reason_code: reasonCode,
    preferences: buildFormPreferences(),
    ...buildExtractorFields("form"),
  });
}

function formatBooking(booking) {
  const date = new Date(booking.start_time);
  const weekday = date.toLocaleDateString(undefined, { weekday: "long" });
  const when = date.toLocaleString();
  if (isFriendlyMode()) {
    return (
      "Your appointment is ready:\n" +
      `${weekday}, ${when}\n` +
      `${prettifyModality(booking.modality)} appointment\n` +
      `Location: ${booking.site}`
    );
  }
  return (
    "Best appointment found:\n" +
    `${weekday}, ${when}\n` +
    `${booking.modality.replace("_", " ")} with ${booking.service_type} (${booking.clinician_id})\n` +
    `${booking.site}`
  );
}

function formatSlotScores(slotScores) {
  if (!slotScores || !slotScores.length) {
    return "No slot scoring returned.";
  }

  return slotScores
    .map((slot) => {
      const dt = new Date(slot.start_time);
      const when = `${dt.toLocaleDateString()} ${dt.toLocaleTimeString()}`;
      if (slot.feasible) {
        return (
          `${slot.slot_number}. ${when} | ${slot.modality} | ` +
          `${slot.service_type} (${slot.clinician_id}) | score=${slot.utility}`
        );
      }
      return (
        `${slot.slot_number}. ${when} | ${slot.modality} | ` +
        `${slot.service_type} (${slot.clinician_id}) | blocked: ${slot.veto_reasons.join(", ")}`
      );
    })
    .join("\n");
}

async function runRouteIfNeeded() {
  if (state.routingDecision) {
    return;
  }

  const routeResponse = await postJson("/api/route", {
    run_id: state.runId,
    intake_summary: state.intakeSummary,
  });

  state.routingDecision = routeResponse.routing_decision;
  state.admin.route = routeResponse;
  updateAdminView();
  updateUserExperience();
}

async function runSchedulingRound(currentPreferences) {
  const offerResponse = await postJson("/api/schedule/offer", {
    run_id: state.runId,
    routing_decision: state.routingDecision,
    preferences: currentPreferences,
  });

  state.admin.scheduling.push(offerResponse);
  state.admin.latestSlotScores = offerResponse.slot_scores || [];
  updateAdminView();
  updateUserExperience();

  const scorePreview = formatSlotScores(offerResponse.slot_scores || []);

  if (offerResponse.status === "booked" && offerResponse.booking) {
    state.preferences = currentPreferences;
    if (isFriendlyMode()) {
      addMessage("assistant", "Assistant", formatBooking(offerResponse.booking));
    } else {
      addMessage("assistant", "Assistant", `${formatBooking(offerResponse.booking)}\n\nSlot Scores:\n${scorePreview}`);
    }
    return;
  }

  if (offerResponse.status === "failed") {
    if (isFriendlyMode()) {
      addMessage("assistant", "Assistant", offerResponse.message || "No suitable appointment found.");
    } else {
      addMessage(
        "assistant",
        "Assistant",
        `${offerResponse.message || "No suitable appointment found."}\n\nSlot Scores:\n${scorePreview}`
      );
    }
    return;
  }

  if (offerResponse.status === "needs_relaxation" && offerResponse.relaxation_questions?.length) {
    if (!isFriendlyMode()) {
      addMessage("assistant", "Assistant", `Current round slot scores:\n${scorePreview}`);
    }
    addRelaxationFormBubble(
      offerResponse.relaxation_questions,
      offerResponse.blocker_summary,
      currentPreferences
    );
    return;
  }

  if (isFriendlyMode()) {
    addMessage("assistant", "Assistant", "No suitable appointment found in this round.");
  } else {
    addMessage("assistant", "Assistant", `Current round slot scores:\n${scorePreview}`);
  }
}

async function previewScheduling(currentPreferences) {
  return postJson("/api/schedule/preview", {
    run_id: state.runId,
    routing_decision: state.routingDecision,
    preferences: currentPreferences,
  });
}

async function handleIntakeResponse(response) {
  setRunId(response.run_id);
  state.admin.intake = response;
  resetAdminRoundData();

  if (response.safety.triggered) {
    addMessage("assistant", "Assistant", response.safety.message || "Safety escalation triggered.");
    state.intakeSummary = null;
    state.preferences = null;
    updateAdminView();
    updateUserExperience();
    return;
  }

  state.intakeSummary = response.intake_summary;
  state.preferences = response.intake_summary ? response.intake_summary.preferences : null;
  const missingFields = response.intake_summary?.missing_fields || [];

  updateAdminView();
  updateUserExperience();

  const questions = response.clarification_questions || [];
  if (questions.length && state.captureMode === "conversation") {
    if (missingFields.length > 0) {
      addClarificationFormBubble(questions);
      updateAdminView();
      updateUserExperience();
      return;
    }

    await runRouteIfNeeded();
    const previewResponse = await previewScheduling(state.preferences);
    state.admin.scheduling.push({ ...previewResponse, preview: true });
    state.admin.latestSlotScores = previewResponse.slot_scores || [];
    updateAdminView();
    updateUserExperience();

    if (previewResponse.status === "booked" && previewResponse.booking) {
      await runSchedulingRound(state.preferences);
      updateAdminView();
      updateUserExperience();
      return;
    }

    await runSchedulingRound(state.preferences);
    updateAdminView();
    updateUserExperience();
    return;
  }

  await runRouteIfNeeded();
  await runSchedulingRound(state.preferences);
  updateAdminView();
  updateUserExperience();
}

async function loadRotaTimetable() {
  try {
    const timetable = await getJson("/api/slots?horizon_days=14");
    state.admin.rotaTimetable = timetable;
    updateAdminView();
  } catch (error) {
    state.admin.rotaTimetable = null;
    adminRotaTimetable.textContent = `Failed to load timetable: ${error.message}`;
  }
}

function resetFormInputs() {
  const userText = document.getElementById("userText");
  const reasonCode = document.getElementById("formReasonCode");
  const preferredModality = document.getElementById("formPreferredModality");

  userText.value = "";
  reasonCode.value = "repeat_prescription";
  preferredModality.value = "";

  formInputs.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.checked = false;
  });

  formHorizonInput.value = "10";
  formSoonestWeightInput.value = "100";
  updateFormSliderLabels();
}

function seedWelcomeMessage() {
  thread.innerHTML = "";
  addMessage(
    "assistant",
    "Assistant",
    "Tell me what you need and I will ask follow-up questions only if they are needed before I look for the best appointment."
  );
}

function resetRunState() {
  const savedRotaTimetable = state.admin.rotaTimetable;

  state.runId = null;
  state.captureMode = "conversation";
  state.humanConversationMode = false;
  state.conversationBaseText = null;
  state.clarificationAnswers = {};
  state.intakeSummary = null;
  state.routingDecision = null;
  state.preferences = null;
  state.admin.intake = null;
  state.admin.route = null;
  state.admin.scheduling = [];
  state.admin.relaxTrail = [];
  state.admin.latestSlotScores = [];
  state.admin.rotaTimetable = savedRotaTimetable;

  humanModeToggle.checked = false;
  setCaptureMode("conversation", { resetConversationBase: false });
  resetFormInputs();
  updateRunIdDisplay();
  updateAdminView();
  updateUserExperience();
  updatePresentationMode();
  seedWelcomeMessage();
}

captureModeSelect.addEventListener("change", () => {
  setCaptureMode(captureModeSelect.value);
});

captureModeConversationButton.addEventListener("click", () => {
  setCaptureMode("conversation");
});

captureModeFormButton.addEventListener("click", () => {
  setCaptureMode("form");
});

userViewToggle.addEventListener("click", () => {
  setViewMode("user");
});

adminViewToggle.addEventListener("click", () => {
  setViewMode("admin");
});

if (auditPageLink) {
  auditPageLink.addEventListener("click", (event) => {
    if (auditPageLink.classList.contains("disabled")) {
      event.preventDefault();
      return;
    }
    event.preventDefault();
    window.location.assign(auditPageLink.href);
  });
}

if (auditJsonLink) {
  auditJsonLink.addEventListener("click", (event) => {
    if (auditJsonLink.classList.contains("disabled")) {
      event.preventDefault();
    }
  });
}

resetButton.addEventListener("click", () => {
  resetRunState();
});

if (humanModeToggle) {
  humanModeToggle.addEventListener("change", () => {
    updatePresentationMode();
  });
}

if (formHorizonInput) {
  formHorizonInput.addEventListener("input", updateFormSliderLabels);
}

if (formSoonestWeightInput) {
  formSoonestWeightInput.addEventListener("input", updateFormSliderLabels);
}

intakeButton.addEventListener("click", async () => {
  try {
    const response =
      state.captureMode === "conversation"
        ? await submitConversationIntake({})
        : await submitFormIntake();
    await handleIntakeResponse(response);
  } catch (error) {
    addMessage("system", "System", error.message);
  }
});

updateRunIdDisplay();
setViewMode("user");
setCaptureMode("conversation", { resetConversationBase: false });
updateFormSliderLabels();
updateAdminView();
updateUserExperience();
seedWelcomeMessage();
loadRotaTimetable();
