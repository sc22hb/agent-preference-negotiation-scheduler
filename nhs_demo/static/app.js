const state = {
  runId: null,
  captureMode: "conversation",
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
  },
};

const thread = document.getElementById("chatThread");
const stateSummary = document.getElementById("stateSummary");
const adminIntake = document.getElementById("adminIntake");
const adminRoute = document.getElementById("adminRoute");
const adminScheduling = document.getElementById("adminScheduling");
const adminRelaxTrail = document.getElementById("adminRelaxTrail");
const adminSlotScores = document.getElementById("adminSlotScores");

const captureModeSelect = document.getElementById("captureMode");
const conversationInputs = document.getElementById("conversationInputs");
const formInputs = document.getElementById("formInputs");

const intakeButton = document.getElementById("btnIntake");

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
    addMessage("user", "Patient", `Clarifications: ${JSON.stringify(answers)}`);

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
  textEl.textContent = "No exact slot yet. Choose which constraints to relax for the next round:";

  const formEl = document.createElement("div");
  formEl.className = "clarify-form";

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
  });

  if (blockerSummary && blockerSummary.ranked_reasons?.length) {
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
    formEl.querySelectorAll("[data-relax-id]").forEach((input) => {
      const key = input.getAttribute("data-relax-id");
      answers[key] = input.value === "true";
    });

    sendButton.disabled = true;
    addMessage("user", "Patient", `Relaxation choices: ${JSON.stringify(answers)}`);

    try {
      const relaxResponse = await postJson("/api/schedule/relax", {
        run_id: state.runId,
        preferences: currentPreferences,
        answers,
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
  adminSlotScores.textContent = state.admin.latestSlotScores.length
    ? JSON.stringify(state.admin.latestSlotScores, null, 2)
    : "No slot scoring yet.";
  adminRelaxTrail.textContent = state.admin.relaxTrail.length
    ? JSON.stringify(state.admin.relaxTrail, null, 2)
    : "No relaxations yet.";
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

function getCheckedValues(fieldName) {
  return Array.from(document.querySelectorAll(`input[name="${fieldName}"]:checked`)).map(
    (input) => input.value
  );
}

function buildFormPreferences() {
  const dayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const preferredModality = document.getElementById("formPreferredModality").value;
  const preferredDaysRaw = getCheckedValues("formPreferredDays");
  const excludedDaysRaw = getCheckedValues("formExcludedDays");
  const excludedPeriod = document.getElementById("formExcludedPeriod").value;
  const dateHorizon = Number(document.getElementById("formHorizon").value || 10);
  const soonestWeight = Number(document.getElementById("formSoonestWeight").value || 60);
  const excludedDaySet = new Set(excludedDaysRaw);
  const preferredDaySet = new Set(preferredDaysRaw.filter((day) => !excludedDaySet.has(day)));
  const preferredDays = dayOrder.filter((day) => preferredDaySet.has(day));
  const excludedDays = dayOrder.filter((day) => excludedDaySet.has(day));

  return {
    preferred_modalities: preferredModality ? [preferredModality] : [],
    excluded_modalities: [],
    preferred_days: preferredDays,
    excluded_days: excludedDays,
    preferred_periods: [],
    excluded_periods: excludedPeriod ? [excludedPeriod] : [],
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
    addMessage("assistant", "Assistant", `${formatBooking(offerResponse.booking)}\n\nSlot Scores:\n${scorePreview}`);
    return;
  }

  if (offerResponse.status === "failed") {
    addMessage(
      "assistant",
      "Assistant",
      `${offerResponse.message || "No suitable appointment found."}\n\nSlot Scores:\n${scorePreview}`
    );
    return;
  }

  addMessage("assistant", "Assistant", `Current round slot scores:\n${scorePreview}`);

  if (offerResponse.status === "needs_relaxation" && offerResponse.relaxation_questions?.length) {
    addRelaxationFormBubble(
      offerResponse.relaxation_questions,
      offerResponse.blocker_summary,
      currentPreferences
    );
    return;
  }

  addMessage("assistant", "Assistant", "No suitable appointment found in this round.");
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

  if (response.extraction_engine === "llm" && response.llm_response) {
    addMessage("assistant", "LLM Parsed JSON", JSON.stringify(response.llm_response, null, 2));
  }

  const questions = response.clarification_questions || [];
  if (questions.length && state.captureMode === "conversation") {
    addClarificationFormBubble(questions);
    updateAdminView();
    return;
  }

  await runRouteIfNeeded();
  await runSchedulingRound(state.preferences);
  updateAdminView();
}

captureModeSelect.addEventListener("change", () => {
  if (state.captureMode !== captureModeSelect.value) {
    state.conversationBaseText = null;
  }
  updateCaptureInputs();
});

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
updateCaptureInputs();
updateAdminView();
