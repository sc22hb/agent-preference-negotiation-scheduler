const state = {
  runId: null,
  captureMode: "conversation",
  humanConversationMode: false,
  conversationBaseText: null,
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

const captureModeSelect = document.getElementById("captureMode");
const humanModeToggle = document.getElementById("humanModeToggle");
const conversationInputs = document.getElementById("conversationInputs");
const formInputs = document.getElementById("formInputs");
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

function addMessage(role, title, content) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  titleEl.textContent = title;

  const bodyEl = document.createElement("div");
  bodyEl.textContent = content;

  bubble.appendChild(titleEl);
  bubble.appendChild(bodyEl);
  thread.appendChild(bubble);
  thread.scrollTop = thread.scrollHeight;
}

function isHumanConversationMode() {
  return Boolean(state.humanConversationMode);
}

function updatePresentationMode() {
  state.humanConversationMode = Boolean(humanModeToggle?.checked);
  document.body.classList.toggle("human-conversation", state.humanConversationMode);
}

function addClarificationFormBubble(questions) {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";

  const titleEl = document.createElement("div");
  titleEl.className = "bubble-title";
  titleEl.textContent = "Assistant";

  const textEl = document.createElement("div");
  textEl.textContent = "I need a couple of clarifications before booking:";

  const formEl = document.createElement("div");
  formEl.className = "clarify-form";

  questions.forEach((q, idx) => {
    const label = document.createElement("label");
    label.className = "clarify-label";
    label.textContent = `${idx + 1}. ${q.prompt}`;

    const input = document.createElement("input");
    input.className = "clarify-input";
    input.setAttribute("data-clarify-id", q.question_id);
    input.placeholder = "Type your answer";

    formEl.appendChild(label);
    formEl.appendChild(input);
  });

  const sendButton = document.createElement("button");
  sendButton.type = "button";
  sendButton.textContent = "Send Answers";

  sendButton.addEventListener("click", async () => {
    const answers = {};
    formEl.querySelectorAll("[data-clarify-id]").forEach((input) => {
      const value = input.value.trim();
      if (value) {
        answers[input.getAttribute("data-clarify-id")] = value;
      }
    });

    if (Object.keys(answers).length === 0) {
      addMessage("system", "System", "Please answer at least one clarification question.");
      return;
    }

    sendButton.disabled = true;
    if (isHumanConversationMode()) {
      addMessage("user", "Patient", Object.values(answers).join("\n"));
    } else {
      addMessage("user", "Patient", `Clarifications: ${JSON.stringify(answers)}`);
    }

    try {
      const response = await submitConversationIntake(answers, true);
      await handleIntakeResponse(response);
      sendButton.textContent = "Answers Sent";
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
  titleEl.textContent = "Assistant";

  const textEl = document.createElement("div");
  textEl.textContent = isHumanConversationMode()
    ? "I could not find an exact slot yet. Would you like to relax any constraints for the next attempt?"
    : "No exact slot yet. Choose which constraints to relax for the next round:";

  const formEl = document.createElement("div");
  formEl.className = "clarify-form";
  const dayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  questions.forEach((q, idx) => {
    const label = document.createElement("label");
    label.className = "clarify-label";
    label.textContent = `${idx + 1}. ${q.prompt}`;

    const select = document.createElement("select");
    select.className = "clarify-input";
    select.setAttribute("data-relax-id", q.key);

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

    if (q.key === "relax_excluded_days") {
      const excludedDays = Array.isArray(currentPreferences?.excluded_days)
        ? currentPreferences.excluded_days
        : [];
      const excludedDayPeriods = Array.isArray(currentPreferences?.excluded_day_periods)
        ? currentPreferences.excluded_day_periods
        : [];
      const dayFromPairs = excludedDayPeriods
        .map((pair) => pair?.day)
        .filter((day) => typeof day === "string");
      const candidateSet = new Set([...excludedDays, ...dayFromPairs]);
      const candidateDays = dayOrder.filter((day) => candidateSet.has(day));

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
          checkbox.setAttribute("data-relax-day-key", q.key);
          checkbox.value = day;

          option.appendChild(checkbox);
          option.appendChild(document.createTextNode(day));
          daySelectWrap.appendChild(option);
        });

        formEl.appendChild(daySelectWrap);
      }
    }
  });

  if (!isHumanConversationMode() && blockerSummary && blockerSummary.ranked_reasons?.length) {
    const blockersEl = document.createElement("div");
    blockersEl.className = "clarify-label";
    blockersEl.textContent = `Main blockers: ${blockerSummary.ranked_reasons.join(", ")}`;
    formEl.appendChild(blockersEl);
  }

  const sendButton = document.createElement("button");
  sendButton.type = "button";
  sendButton.textContent = "Apply and Retry";

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
    if (isHumanConversationMode()) {
      const approved = Object.entries(answers)
        .filter((entry) => entry[1])
        .map((entry) => entry[0]);
      const message = approved.length
        ? `Yes, please relax: ${approved.join(", ")}`
        : "No, keep my current constraints.";
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

      await runSchedulingRound(state.preferences);
      sendButton.textContent = "Applied";
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

function setRunId(runId) {
  state.runId = runId;
  document.getElementById("runIdView").textContent = `Run ID: ${runId}`;

  const auditPage = document.getElementById("auditPageLink");
  const auditJson = document.getElementById("auditJsonLink");
  auditPage.href = `/audit/${runId}`;
  auditJson.href = `/api/run/${runId}/audit`;
  auditPage.classList.remove("disabled");
  auditJson.classList.remove("disabled");
}

function updateCaptureInputs() {
  state.captureMode = captureModeSelect.value;
  conversationInputs.style.display = state.captureMode === "conversation" ? "block" : "none";
  formInputs.style.display = state.captureMode === "form" ? "block" : "none";
  updateAdminView();
}

function updateAdminView() {
  stateSummary.textContent = JSON.stringify(
    {
      run_id: state.runId,
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
  for (const service of serviceKeys) {
    const slots = payload.services[service] || [];
    lines.push("");
    lines.push(`${service} (${slots.length} slots)`);
    slots.forEach((slot, index) => {
      const dt = new Date(slot.start_time);
      const weekday = dt.toLocaleDateString(undefined, { weekday: "short" });
      const when = `${weekday} ${dt.toLocaleDateString()} ${dt.toLocaleTimeString()}`;
      lines.push(
        `${index + 1}. ${when} | ${slot.modality} | ${slot.clinician_id} | ${slot.slot_id}`
      );
    });
  }

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
    const err = await response.json();
    throw new Error(err.detail || response.statusText);
  }
  return response.json();
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || response.statusText);
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
  const dayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri"];
  const periodOrder = ["morning", "afternoon", "evening"];
  const preferredModality = document.getElementById("formPreferredModality").value;
  const modalityOrder = ["in_person", "phone", "video"];
  const excludedModalities = modalityOrder.filter((modality) =>
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

  for (const day of dayOrder) {
    const preferredDayChecked = Boolean(document.getElementById(`pref-day-${day}`)?.checked);
    const hardDayChecked = Boolean(document.getElementById(`hard-day-${day}`)?.checked);

    if (hardDayChecked) {
      excludedDays.push(day);
    } else if (preferredDayChecked) {
      preferredDays.push(day);
    }

    for (const period of periodOrder) {
      const preferredPeriodChecked = Boolean(document.getElementById(`pref-slot-${day}-${period}`)?.checked);
      const hardPeriodChecked = Boolean(document.getElementById(`hard-slot-${day}-${period}`)?.checked);

      if (hardDayChecked || hardPeriodChecked) {
        if (!hardDayChecked && hardPeriodChecked) {
          excludedDayPeriods.push({ day, period });
        }
        continue;
      }

      if (preferredPeriodChecked) {
        preferredDayPeriods.push({ day, period });
      }
    }
  }

  return {
    preferred_modalities: preferredModalities,
    excluded_modalities: excludedModalities,
    preferred_days: preferredDays,
    excluded_days: excludedDays,
    // Keep period preferences day-specific via preferred_day_periods to avoid over-generalizing.
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
  }

  if (!isFollowUp) {
    addMessage("user", "Patient", state.conversationBaseText);
  }

  if (isFollowUp) {
    if (!state.runId) {
      throw new Error("Missing run id for intake refinement.");
    }
    return postJson("/api/intake/refine", {
      run_id: state.runId,
      clarification_answers: clarificationAnswers,
      ...buildExtractorFields("conversation"),
    });
  }

  return postJson("/api/intake", {
    user_text: state.conversationBaseText,
    clarification_answers: clarificationAnswers,
    ...buildExtractorFields("conversation"),
  });
}

async function submitFormIntake() {
  const reasonCode = document.getElementById("formReasonCode").value;
  if (!reasonCode) {
    throw new Error("Please choose a reason first.");
  }

  addMessage("user", "Patient", "Submitted form preferences.");
  return postJson("/api/intake/form", {
    reason_code: reasonCode,
    preferences: buildFormPreferences(),
    ...buildExtractorFields("form"),
  });
}

function formatBooking(booking) {
  const dt = new Date(booking.start_time);
  const weekday = dt.toLocaleDateString(undefined, { weekday: "long" });
  const when = dt.toLocaleString();
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

  const lines = slotScores.map((slot) => {
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
  });

  return lines.join("\n");
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

  const scorePreview = formatSlotScores(offerResponse.slot_scores || []);

  if (offerResponse.status === "booked" && offerResponse.booking) {
    state.preferences = currentPreferences;
    if (isHumanConversationMode()) {
      addMessage("assistant", "Assistant", formatBooking(offerResponse.booking));
    } else {
      addMessage("assistant", "Assistant", `${formatBooking(offerResponse.booking)}\n\nSlot Scores:\n${scorePreview}`);
    }
    return;
  }

  if (offerResponse.status === "failed") {
    if (isHumanConversationMode()) {
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
    if (!isHumanConversationMode()) {
      addMessage("assistant", "Assistant", `Current round slot scores:\n${scorePreview}`);
    }
    addRelaxationFormBubble(
      offerResponse.relaxation_questions,
      offerResponse.blocker_summary,
      currentPreferences
    );
    return;
  }

  if (isHumanConversationMode()) {
    addMessage("assistant", "Assistant", "No suitable appointment found in this round.");
  } else {
    addMessage("assistant", "Assistant", `Current round slot scores:\n${scorePreview}`);
  }

  return offerResponse;
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
    return;
  }

  state.intakeSummary = response.intake_summary;
  state.preferences = response.intake_summary ? response.intake_summary.preferences : null;
  const missingFields = response.intake_summary?.missing_fields || [];

  if (!isHumanConversationMode() && response.extraction_engine === "llm" && response.llm_response) {
    addMessage("assistant", "LLM Parsed JSON", JSON.stringify(response.llm_response, null, 2));
  }

  const questions = response.clarification_questions || [];
  if (questions.length && state.captureMode === "conversation") {
    // Required missing fields must be clarified before booking.
    if (missingFields.length > 0) {
      addClarificationFormBubble(questions);
      updateAdminView();
      return;
    }

    await runRouteIfNeeded();
    const previewResponse = await previewScheduling(state.preferences);
    state.admin.scheduling.push({ ...previewResponse, preview: true });
    state.admin.latestSlotScores = previewResponse.slot_scores || [];
    updateAdminView();

    if (previewResponse.status === "booked" && previewResponse.booking) {
      await runSchedulingRound(state.preferences);
      updateAdminView();
      return;
    }

    addClarificationFormBubble(questions);
    updateAdminView();
    return;
  }

  await runRouteIfNeeded();
  await runSchedulingRound(state.preferences);
  updateAdminView();
}

async function loadRotaTimetable() {
  try {
    // Show next two weeks in admin while keeping DB shared and larger in backend.
    const timetable = await getJson("/api/slots?horizon_days=14");
    state.admin.rotaTimetable = timetable;
    updateAdminView();
  } catch (error) {
    state.admin.rotaTimetable = null;
    if (adminRotaTimetable) {
      adminRotaTimetable.textContent = `Failed to load timetable: ${error.message}`;
    }
  }
}

captureModeSelect.addEventListener("change", () => {
  if (state.captureMode !== captureModeSelect.value) {
    state.conversationBaseText = null;
  }
  updateCaptureInputs();
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

addMessage(
  "assistant",
  "Assistant",
  "Tell me your request and I will ask clarifications if needed, then return the best appointment."
);
updatePresentationMode();
updateCaptureInputs();
updateFormSliderLabels();
updateAdminView();
loadRotaTimetable();
