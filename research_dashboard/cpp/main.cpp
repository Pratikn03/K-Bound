#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string iso8601_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string history_filename_stamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%dT%H%M%SZ");
    return oss.str();
}

json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open JSON: " + path.string());
    }
    json doc;
    in >> doc;
    return doc;
}

json json_or_null(const json& object, const std::string& key) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return *it;
}

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

fs::path find_repo_root(fs::path start) {
    start = fs::absolute(start);
    while (true) {
        const fs::path marker = start / "elara_master_c" / "audits" / "checklist_progress.json";
        if (fs::exists(marker)) {
            return start;
        }
        if (!start.has_parent_path() || start.parent_path() == start) {
            break;
        }
        start = start.parent_path();
    }
    return {};
}

std::string yaml_scalar(const std::string& text, const std::string& key) {
    const std::regex re("^\\s*" + key + "\\s*:\\s*(.+?)\\s*$", std::regex::icase);
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        std::smatch match;
        if (std::regex_match(line, match, re) && match.size() > 1) {
            std::string value = match[1].str();
            if (!value.empty() && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            if (!value.empty() && value.front() == '#') {
                continue;
            }
            const auto hash = value.find('#');
            if (hash != std::string::npos) {
                value = value.substr(0, hash);
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
                value.pop_back();
            }
            return value;
        }
    }
    return {};
}

json parse_protocol_yaml(const fs::path& path) {
    json out = json::object();
    out["path"] = "research_lock/FLAGSHIP_DEV_PROTOCOL_v1.yaml";
    if (!fs::exists(path)) {
        out["found"] = false;
        return out;
    }
    const std::string text = read_text_file(path);
    out["found"] = true;
    out["version"] = yaml_scalar(text, "version");
    out["status"] = yaml_scalar(text, "status");
    out["ratified"] = yaml_scalar(text, "ratified");
    out["confirmatory_blocked"] = text.find("blocked_until_stop_rule: true") != std::string::npos;
    return out;
}

std::vector<std::string> gate_evidence_files(const std::string& gate_id) {
    static const std::map<std::string, std::vector<std::string>> kMap = {
        {"gate_a", {"elara_master_c/audits/gate_a_expert_qualification_v2.json"}},
        {"gate_b", {"experiments/fusion/master_c_real_domain_results.json",
                    "experiments/fusion/master_c_mvtec_supervised_paired_results.json"}},
        {"gate_c", {"experiments/phase2"}},
        {"gate_d", {"elara_master_c/audits/confirmatory_statistics_report.json"}},
        {"gate_e", {"elara_master_c/audits/confirmatory_statistics_report.json",
                    "experiments/fusion/m2_external_3d_adam_transfer_v1_confirmatory_results.json"}},
        {"gate_e_bounded_v3", {"elara_master_c/audits/confirmatory_statistics_report.json",
                               "research_lock/SCENARIO_C_V3_INTEGRATION_v1.yaml"}},
        {"gate_e_positive_transfer", {"elara_master_c/audits/confirmatory_statistics_report.json",
                                      "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml",
                                      "elara_master_c/audits/positive_transfer_development_report.json"}},
        {"gate_f", {"elara_master_c/audits/checklist_progress.json"}},
        {"gate_f_scientific", {"elara_master_c/audits/confirmatory_statistics_report.json",
                               "elara_master_c/audits/checklist_progress.json"}},
        {"gate_f_bounded_v3", {"elara_master_c/audits/confirmatory_statistics_report.json",
                               "research_lock/SCENARIO_C_V3_INTEGRATION_v1.yaml"}},
        {"gate_f_positive_transfer", {"elara_master_c/audits/confirmatory_statistics_report.json",
                                      "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml"}},
    };
    const auto it = kMap.find(gate_id);
    if (it == kMap.end()) {
        return {};
    }
    return it->second;
}

json gate_entry(const json& items, const std::string& gate_id) {
    json base = json{{"id", gate_id}, {"done", false}, {"description", "missing from checklist"}};
    for (const auto& item : items) {
        if (item.value("id", std::string{}) == gate_id) {
            base = {
                {"id", gate_id},
                {"description", item.value("description", std::string{})},
                {"done", item.value("done", false)},
                {"stage", item.value("stage", std::string{})},
                {"evidence", item.value("evidence", std::string{})},
            };
            break;
        }
    }
    json paths = json::array();
    for (const auto& rel : gate_evidence_files(gate_id)) {
        paths.push_back(rel);
    }
    base["evidence_paths"] = paths;
    return base;
}

json summarize_cell(const json& cell) {
    json ci = json::object();
    if (cell.contains("bootstrap_95_ci") && cell["bootstrap_95_ci"].is_object()) {
        ci = cell["bootstrap_95_ci"];
    }
    return {
        {"family", cell.value("family", std::string{})},
        {"benchmark", cell.value("benchmark", std::string{})},
        {"protocol", cell.value("protocol", std::string{})},
        {"results_path", cell.value("results_path", std::string{})},
        {"n_seeds", cell.value("n_seeds", 0)},
        {"mean_rga_auc", cell.value("mean_rga_auc", 0.0)},
        {"mean_base_auc", cell.value("mean_base_auc", 0.0)},
        {"mean_delta_roc_auc", cell.value("mean_delta_roc_auc", 0.0)},
        {"bootstrap_95_ci", ci},
        {"cell_valid", cell.value("cell_valid", false)},
        {"validity_reasons", cell.value("validity_reasons", json::array())},
        {"gate_d_pass", cell.value("gate_d_pass", false)},
        {"gate_e_pass", cell.value("gate_e_pass", false)},
        {"t5_confirmatory_pass", cell.value("t5_confirmatory_pass", false)},
    };
}

json items_by_stage(const json& items) {
    json stages = json::object();
    for (const auto& item : items) {
        const std::string stage = item.value("stage", "UNKNOWN");
        if (!stages.contains(stage)) {
            stages[stage] = json{{"done", 0}, {"total", 0}, {"items", json::array()}};
        }
        stages[stage]["total"] = stages[stage]["total"].get<int>() + 1;
        if (item.value("done", false)) {
            stages[stage]["done"] = stages[stage]["done"].get<int>() + 1;
        }
        stages[stage]["items"].push_back({
            {"id", item.value("id", std::string{})},
            {"description", item.value("description", std::string{})},
            {"done", item.value("done", false)},
        });
    }
    for (auto it = stages.begin(); it != stages.end(); ++it) {
        const int done = it.value()["done"].get<int>();
        const int total = it.value()["total"].get<int>();
        it.value()["percent"] = total > 0 ? std::round(1000.0 * done / total) / 10.0 : 0.0;
    }
    return stages;
}

double nested_double(const json& snap, const char* a, const char* b, double fallback) {
    if (!snap.contains(a) || !snap.at(a).contains(b)) {
        return fallback;
    }
    return snap.at(a).at(b).get<double>();
}

bool nested_bool(const json& snap, const char* a, const char* b, bool fallback) {
    if (!snap.contains(a) || !snap.at(a).contains(b)) {
        return fallback;
    }
    return snap.at(a).at(b).get<bool>();
}

json timeline_point(const json& snap) {
    json point = {
        {"generated_at", snap.value("generated_at", std::string{})},
        {"percent_complete", nested_double(snap, "checklist", "percent_complete", 0.0)},
        {"execution_percent", nested_double(snap, "checklist", "execution_percent", 0.0)},
        {"scientific_ready", nested_bool(snap, "checklist", "scientific_scenario_c_ready", false)},
        {"gate_d_m1", nested_bool(snap, "confirmatory", "gate_d_m1", false)},
        {"gate_e_m2", nested_bool(snap, "confirmatory", "gate_e_m2_transfer_confirmed", false)},
        {"gate_e_positive_transfer", nested_bool(snap, "confirmatory", "gate_e_positive_transfer_confirmed", false)},
        {"blocker_count", snap.value("blockers", json::array()).size()},
    };
    if (snap.contains("confirmatory") && snap["confirmatory"].contains("cells")) {
        for (const auto& cell : snap["confirmatory"]["cells"]) {
            const std::string fam = cell.value("family", std::string{});
            if (!fam.empty()) {
                point["delta_" + fam] = cell.value("mean_delta_roc_auc", 0.0);
            }
        }
    }
    return point;
}

json compute_diff(const json& prev, const json& curr) {
    json diff = json::object();
    if (prev.is_null() || prev.empty()) {
        diff["has_previous"] = false;
        return diff;
    }
    diff["has_previous"] = true;
    diff["previous_at"] = prev.value("generated_at", std::string{});
    diff["current_at"] = curr.value("generated_at", std::string{});

    const double prev_pct = nested_double(prev, "checklist", "percent_complete", 0.0);
    const double curr_pct = nested_double(curr, "checklist", "percent_complete", 0.0);
    diff["checklist_percent_delta"] = std::round((curr_pct - prev_pct) * 10.0) / 10.0;

    const double prev_exec = nested_double(prev, "checklist", "execution_percent", 0.0);
    const double curr_exec = nested_double(curr, "checklist", "execution_percent", 0.0);
    diff["execution_percent_delta"] = std::round((curr_exec - prev_exec) * 10.0) / 10.0;

    json gate_changes = json::array();
    const json& prev_gates = prev.value("gates", json::object());
    const json& curr_gates = curr.value("gates", json::object());
    for (auto it = curr_gates.begin(); it != curr_gates.end(); ++it) {
        const std::string id = it.key();
        const bool was = prev_gates.value(id, json::object()).value("done", false);
        const bool now = it.value().value("done", false);
        if (was != now) {
            gate_changes.push_back({
                {"id", id},
                {"from", was},
                {"to", now},
            });
        }
    }
    diff["gate_changes"] = gate_changes;

    auto blocker_ids = [](const json& snap) {
        std::vector<std::string> ids;
        for (const auto& b : snap.value("blockers", json::array())) {
            ids.push_back(b.value("id", std::string{}));
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    };
    const auto prev_b = blocker_ids(prev);
    const auto curr_b = blocker_ids(curr);
    json added = json::array();
    json removed = json::array();
    for (const auto& id : curr_b) {
        if (!std::binary_search(prev_b.begin(), prev_b.end(), id)) {
            added.push_back(id);
        }
    }
    for (const auto& id : prev_b) {
        if (!std::binary_search(curr_b.begin(), curr_b.end(), id)) {
            removed.push_back(id);
        }
    }
    diff["blockers_added"] = added;
    diff["blockers_removed"] = removed;

    json cell_deltas = json::array();
    std::map<std::string, double> prev_cells;
    const json prev_cells_arr =
        prev.contains("confirmatory") ? prev.at("confirmatory").value("cells", json::array()) : json::array();
    const json curr_cells_arr =
        curr.contains("confirmatory") ? curr.at("confirmatory").value("cells", json::array()) : json::array();
    for (const auto& c : prev_cells_arr) {
        prev_cells[c.value("family", std::string{})] = c.value("mean_delta_roc_auc", 0.0);
    }
    for (const auto& c : curr_cells_arr) {
        const std::string fam = c.value("family", std::string{});
        const double now = c.value("mean_delta_roc_auc", 0.0);
        if (prev_cells.count(fam)) {
            const double delta = now - prev_cells[fam];
            if (std::abs(delta) > 1e-9) {
                cell_deltas.push_back({
                    {"family", fam},
                    {"previous_delta", prev_cells[fam]},
                    {"current_delta", now},
                    {"change", delta},
                });
            }
        }
    }
    diff["confirmatory_cell_changes"] = cell_deltas;
    return diff;
}

std::vector<fs::path> list_history_files(const fs::path& history_dir, size_t max_files) {
    std::vector<fs::path> files;
    if (!fs::exists(history_dir)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(history_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.rfind("snapshot_", 0) == 0 && entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.size() > max_files) {
        files.erase(files.begin(), files.end() - max_files);
    }
    return files;
}

json build_history_timeline(const fs::path& history_dir, size_t max_points) {
    json timeline = json::array();
    for (const auto& path : list_history_files(history_dir, max_points)) {
        try {
            timeline.push_back(timeline_point(read_json_file(path)));
        } catch (const std::exception& ex) {
            std::cerr << "Skip history " << path << ": " << ex.what() << "\n";
        }
    }
    return timeline;
}

fs::path find_latest_history(const fs::path& history_dir, const fs::path& exclude) {
    fs::path latest;
    const std::string exclude_str = exclude.empty() ? std::string{} : fs::absolute(exclude).string();
    for (const auto& path : list_history_files(history_dir, 500)) {
        if (!exclude_str.empty() && fs::absolute(path).string() == exclude_str) {
            continue;
        }
        latest = path;
    }
    return latest;
}

void prune_history(const fs::path& history_dir, size_t keep) {
    auto files = list_history_files(history_dir, 10000);
    if (files.size() <= keep) {
        return;
    }
    const size_t remove_count = files.size() - keep;
    for (size_t i = 0; i < remove_count; ++i) {
        std::error_code ec;
        fs::remove(files[i], ec);
    }
}

bool item_done(const json& items, const std::string& id) {
    for (const auto& item : items) {
        if (item.value("id", std::string{}) == id) {
            return item.value("done", false);
        }
    }
    return false;
}

json load_flagship_stop_rule(const fs::path& audits) {
    json out = {{"found", false}, {"passed", false}, {"min_delta_vs_sar", 0.01}, {"best_mean_delta", 0.0}};
    const fs::path sweep = audits / "flagship_val_sweep_m2_validation.json";
    if (!fs::exists(sweep)) {
        return out;
    }
    try {
        const json doc = read_json_file(sweep);
        out["found"] = true;
        out["passed"] = doc.value("stop_rule_passed", false);
        out["min_delta_vs_sar"] = doc.value("min_delta_vs_sar", 0.01);
        if (doc.contains("winner") && doc["winner"].is_object()) {
            out["best_mean_delta"] = doc["winner"].value("delta_vs_sar_mean", 0.0);
            out["winner"] = doc["winner"].value("name", std::string{});
        }
    } catch (...) {
    }
    return out;
}

json build_claim_status(
    const json& items,
    const json& summary,
    const json& confirmatory,
    const json& gates,
    const json& flagship_stop) {
    const bool gate_a = gates.value("gate_a", json::object()).value("done", false);
    const bool gate_b = gates.value("gate_b", json::object()).value("done", false);
    const bool gate_c = gates.value("gate_c", json::object()).value("done", false);
    const bool gate_d = confirmatory.value("gate_d_m1", false);
    const bool gate_e = confirmatory.value("gate_e_m2_transfer_confirmed", false);
    const bool gate_e_strict = confirmatory.value("gate_e_m2_transfer_confirmed_strict", gate_e);
    const bool bounded_v3 = confirmatory.value("gate_e_m2_bounded_v3_pass", false);
    const bool positive_transfer = confirmatory.value("gate_e_positive_transfer_confirmed", false);
    const std::string positive_transfer_status =
        confirmatory.value("gate_e_positive_transfer_status", std::string{});
    const bool positive_transfer_official_fail = positive_transfer_status == "OFFICIAL_FAIL";
    const bool gate_f_sci = confirmatory.value("gate_f_scenario_c_scientific", false);
    const bool gate_f_bounded = confirmatory.value("gate_f_bounded_v3", false);
    const bool gate_f_positive = confirmatory.value("gate_f_positive_transfer_track", false);
    const bool t5_m1 = confirmatory.value("t5_m1", false);
    const bool t6 = item_done(items, "t6_gdr_audit");
    const bool pred_arch = item_done(items, "pred_archive_all_runs");
    const bool governance = item_done(items, "t0_governance_validator");

    auto pillar = [](const std::string& id, const std::string& name, const std::string& req,
                     const std::string& status, bool pass) {
        return json{
            {"id", id},
            {"name", name},
            {"required_evidence", req},
            {"status", status},
            {"pass", pass},
        };
    };

    json pillars = json::array({
        pillar("P1", "Mechanism validity", "Base RGA under degradation; quiet on clean",
               gate_c ? "PARTIAL" : "NOT ESTABLISHED", gate_c),
        pillar("P2", "Strong-baseline superiority", "Beat frozen SAR on confirmatory M1",
               (gate_d && t5_m1) ? "PASS" : "NOT ESTABLISHED", gate_d && t5_m1),
        pillar("P3", "Multimodal generalization", "Multiple naturally co-observed datasets",
               (gate_a && gate_b) ? "PARTIAL" : "NOT ESTABLISHED", false),
        pillar("P4", "Held-out transfer", "Strict clean M2 external positive CI vs SAR",
               gate_e_strict ? "PASS"
                             : (positive_transfer ? "NATURAL SAR+CW PASS"
                                                  : (positive_transfer_official_fail
                                                         ? "D13 OFFICIAL FAIL"
                                                         : (bounded_v3 ? "BOUNDED STRESS ONLY"
                                                                       : "NOT CONFIRMED"))),
               gate_e_strict || positive_transfer),
        pillar("P5", "Theory & certificate", "GDR + switching conditions",
               t6 ? "PARTIAL" : "NOT ESTABLISHED", false),
        pillar("P6", "Deployment auditability", "Archives + monitoring path",
               pred_arch ? "PARTIAL" : "NOT ESTABLISHED", false),
    });

    const int pillars_pass = static_cast<int>(std::count_if(pillars.begin(), pillars.end(),
        [](const json& p) { return p.value("pass", false); }));

    std::string tier = "tier_1_bounded";
    if (gate_f_sci) {
        tier = "tier_3_flagship";
    } else if (gate_f_positive || positive_transfer) {
        tier = "tier_2_natural_positive_transfer";
    } else if (gate_f_bounded || bounded_v3) {
        tier = "tier_1_bounded";
    }

    std::string one_sentence;
    if (gate_f_sci) {
        one_sentence =
            "ELARA confirms superiority over frozen SAR on M1 and M2 external confirmatory endpoints.";
    } else if (positive_transfer) {
        one_sentence =
            "ELARA has a D13 natural positive-transfer confirmation against both SAR and CW, while legacy strict Gate E remains preserved separately.";
    } else if (positive_transfer_official_fail && bounded_v3) {
        one_sentence =
            "ELARA has bounded v3 stress evidence, but the fresh Real-IAD D13 natural-transfer attempt beat CW and failed the SAR endpoint, so strict clean transfer remains not confirmed.";
    } else if (bounded_v3) {
        one_sentence =
            "ELARA has a bounded v3 result: M1 in-domain superiority plus stress-regime "
            "transfer evidence, while strict clean external Gate E remains not confirmed.";
    } else {
        one_sentence =
            "ELARA shows bounded mechanism and pipeline evidence, but does not confirm superiority "
            "over frozen SAR on M2 external (Δ ≈ −0.04, paired p < 0.01).";
    }

    json tier1 = json::array({
        json{{"id", "evidence_frozen"}, {"label", "Evidence frozen & auditable"}, {"done", governance}},
        json{{"id", "comparator_frozen"}, {"label", "Comparator frozen (D5)"},
             {"done", item_done(items, "t3_strong_baselines")}},
        json{{"id", "archives"}, {"label", "Prediction archives"}, {"done", pred_arch}},
        json{{"id", "gate_c_mechanism"}, {"label", "Base RGA mechanism"}, {"done", gate_c}},
    });
    int tier1_done = 0;
    for (const auto& t : tier1) {
        if (t.value("done", false)) {
            ++tier1_done;
        }
    }

    json tier2 = json::array({
        json{{"id", "m2_transfer_strict"}, {"label", "Strict M2 clean transfer confirmed"},
             {"done", gate_e_strict}},
        json{{"id", "bounded_v3_stress"}, {"label", "Bounded v3 stress evidence"},
             {"done", bounded_v3}},
        json{{"id", "positive_transfer"}, {"label", "D13 natural SAR+CW transfer"},
             {"done", positive_transfer}},
        json{{"id", "gate_d_m1"}, {"label", "M1 superiority vs SAR"}, {"done", gate_d && t5_m1}},
        json{{"id", "gate_a_v2"}, {"label", "Upstream experts (Gate A v2)"}, {"done", gate_a}},
    });
    int tier2_done = 0;
    for (const auto& t : tier2) {
        if (t.value("done", false)) {
            ++tier2_done;
        }
    }

    json tier3 = json::array({
        json{{"id", "gate_f_sci"}, {"label", "Scenario C scientific ready"}, {"done", gate_f_sci}},
        json{{"id", "flagship_stop"}, {"label", "Flagship val stop rule (Δ≥0.01)"},
             {"done", flagship_stop.value("passed", false)}},
        json{{"id", "all_gates_de"}, {"label", "Gate D + natural transfer pass"},
             {"done", gate_d && (gate_e_strict || positive_transfer)}},
    });
    int tier3_done = 0;
    for (const auto& t : tier3) {
        if (t.value("done", false)) {
            ++tier3_done;
        }
    }

    return {
        {"readiness_tier", tier},
        {"scientific_ready", gate_f_sci},
        {"strict_scientific_ready", gate_f_sci},
        {"bounded_v3_evidence_ready", bounded_v3},
        {"positive_transfer_confirmed", positive_transfer},
        {"execution_ready", summary.value("execution_percent", 0.0) >= 94.0},
        {"one_sentence_claim", one_sentence},
        {"central_claim_ratified", false},
        {"pillars", pillars},
        {"pillars_pass_count", pillars_pass},
        {"pillars_total", 6},
        {"readiness_checklists", {
            {"tier1_bounded_paper", {{"done", tier1_done}, {"total", tier1.size()}, {"items", tier1}}},
            {"tier2_generalization", {{"done", tier2_done}, {"total", tier2.size()}, {"items", tier2}}},
            {"tier3_flagship", {{"done", tier3_done}, {"total", tier3.size()}, {"items", tier3}}},
        }},
        {"flagship_stop_rule", flagship_stop},
        {"contract_path", "research_lock/SCENARIO_C_CLAIM_CONTRACT.md"},
    };
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --repo-root PATH       AutoML_Flagship_V8 root\n"
        << "  --output PATH          snapshot.json (default: research_dashboard/web/data/snapshot.json)\n"
        << "  --history-dir PATH     history folder (default: web/data/history)\n"
        << "  --max-history N        keep N history files (default: 100)\n"
        << "  --no-history           skip writing history archive\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    fs::path repo_root;
    fs::path output_path;
    fs::path history_dir;
    size_t max_history = 100;
    bool write_history = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--repo-root" && i + 1 < argc) {
            repo_root = fs::path(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = fs::path(argv[++i]);
        } else if (arg == "--history-dir" && i + 1 < argc) {
            history_dir = fs::path(argv[++i]);
        } else if (arg == "--max-history" && i + 1 < argc) {
            max_history = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--no-history") {
            write_history = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (repo_root.empty()) {
        repo_root = find_repo_root(fs::current_path());
    }
    if (repo_root.empty()) {
        std::cerr << "Could not locate repo root. Pass --repo-root.\n";
        return 1;
    }
    repo_root = fs::absolute(repo_root);

    if (output_path.empty()) {
        output_path = repo_root / "research_dashboard" / "web" / "data" / "snapshot.json";
    }
    if (history_dir.empty()) {
        history_dir = output_path.parent_path() / "history";
    }

    const fs::path audits = repo_root / "elara_master_c" / "audits";
    const json checklist = read_json_file(audits / "checklist_progress.json");
    const json confirmatory = read_json_file(audits / "confirmatory_statistics_report.json");

    json catalog_summary = json::object();
    const fs::path catalog_path = audits / "python_file_catalog.json";
    if (fs::exists(catalog_path)) {
        const json catalog = read_json_file(catalog_path);
        catalog_summary = {
            {"total", catalog.value("total", 0)},
            {"used_count", catalog.value("used_count", 0)},
            {"unused_count", catalog.value("unused_count", 0)},
            {"by_category", catalog.value("by_category", json::object())},
        };
    } else {
        catalog_summary = {{"found", false}};
    }

    const json& summary = checklist.at("summary");
    const json& items = checklist.at("items");

    const std::vector<std::string> gate_ids = {
        "gate_a", "gate_b", "gate_c", "gate_d", "gate_e", "gate_e_bounded_v3",
        "gate_e_positive_transfer", "gate_f", "gate_f_scientific",
        "gate_f_bounded_v3", "gate_f_positive_transfer",
    };
    json gates = json::object();
    for (const auto& gate_id : gate_ids) {
        gates[gate_id] = gate_entry(items, gate_id);
    }

    json cells = json::array();
    if (confirmatory.contains("cells") && confirmatory["cells"].is_array()) {
        for (const auto& cell : confirmatory["cells"]) {
            cells.push_back(summarize_cell(cell));
        }
    }

    const json flagship_stop = load_flagship_stop_rule(audits);

    json raw_sources = {
        {"checklist_progress", "elara_master_c/audits/checklist_progress.json"},
        {"confirmatory_statistics", "elara_master_c/audits/confirmatory_statistics_report.json"},
        {"python_file_catalog", "elara_master_c/audits/python_file_catalog.json"},
        {"flagship_protocol", "research_lock/FLAGSHIP_DEV_PROTOCOL_v1.yaml"},
        {"claim_contract", "research_lock/SCENARIO_C_CLAIM_CONTRACT.md"},
        {"positive_transfer_protocol", "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml"},
        {"positive_transfer_development", "elara_master_c/audits/positive_transfer_development_report.json"},
        {"positive_transfer_confirmatory", "experiments/fusion/positive_transfer_confirmatory_result.json"},
    };

    json snapshot = {
        {"generated_at", iso8601_now()},
        {"repo_root", repo_root.string()},
        {"sources", json::array({
            "elara_master_c/audits/checklist_progress.json",
            "elara_master_c/audits/confirmatory_statistics_report.json",
            "elara_master_c/audits/python_file_catalog.json",
            "research_lock/FLAGSHIP_DEV_PROTOCOL_v1.yaml",
            "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml",
        })},
        {"raw_sources", raw_sources},
        {"checklist", {
            {"done", summary.value("done", 0)},
            {"total", summary.value("total", 0)},
            {"percent_complete", summary.value("percent_complete", 0.0)},
            {"execution_percent", summary.value("execution_percent", 0.0)},
            {"execution_complete", summary.value("execution_complete", false)},
            {"scientific_scenario_c_ready", summary.value("scientific_scenario_c_ready", false)},
            {"m2_transfer_confirmed", summary.value("m2_transfer_confirmed", false)},
            {"bounded_v3_evidence_ready", summary.value("bounded_v3_evidence_ready", false)},
            {"gate_f_bounded_v3", summary.value("gate_f_bounded_v3", false)},
            {"positive_transfer_confirmed", summary.value("positive_transfer_confirmed", false)},
            {"gate_f_positive_transfer", summary.value("gate_f_positive_transfer", false)},
            {"verdict", summary.value("verdict", std::string{})},
            {"items_by_stage", items_by_stage(items)},
            {"remaining_count", summary.value("remaining_blockers", json::array()).size()},
        }},
        {"gates", gates},
        {"confirmatory", {
            {"gate_d_m1", confirmatory.value("gate_d_m1", false)},
            {"gate_d_m2", confirmatory.value("gate_d_m2", false)},
            {"gate_d_m2_external", confirmatory.value("gate_d_m2_external", false)},
            {"gate_d_m2_proxy", confirmatory.value("gate_d_m2_proxy", false)},
            {"gate_e_m2_transfer_confirmed", confirmatory.value("gate_e_m2_transfer_confirmed", false)},
            {"gate_e_m2_bounded_v3_pass", confirmatory.value("gate_e_m2_bounded_v3_pass", false)},
            {"gate_e_m2_checklist_pass", confirmatory.value("gate_e_m2_checklist_pass", false)},
            {"gate_e_m2_transfer_confirmed_strict",
             confirmatory.value("gate_e_m2_transfer_confirmed_strict", false)},
            {"gate_e_positive_transfer_confirmed",
             confirmatory.value("gate_e_positive_transfer_confirmed", false)},
            {"gate_e_positive_transfer_official",
             confirmatory.value("gate_e_positive_transfer_official", false)},
            {"gate_e_positive_transfer_official_attempt",
             confirmatory.value("gate_e_positive_transfer_official_attempt", false)},
            {"gate_e_positive_transfer_status",
             confirmatory.value("gate_e_positive_transfer_status", std::string{"PENDING_FRESH_HOLDOUT"})},
            {"gate_e_positive_transfer_delta_vs_sar",
             json_or_null(confirmatory, "gate_e_positive_transfer_delta_vs_sar")},
            {"gate_e_positive_transfer_delta_vs_cw",
             json_or_null(confirmatory, "gate_e_positive_transfer_delta_vs_cw")},
            {"gate_e_positive_transfer_ci95_vs_sar",
             json_or_null(confirmatory, "gate_e_positive_transfer_ci95_vs_sar")},
            {"gate_e_positive_transfer_ci95_vs_cw",
             json_or_null(confirmatory, "gate_e_positive_transfer_ci95_vs_cw")},
            {"gate_e_positive_transfer_vs_sar_pass",
             confirmatory.value("gate_e_positive_transfer_vs_sar_pass", false)},
            {"gate_e_positive_transfer_vs_cw_pass",
             confirmatory.value("gate_e_positive_transfer_vs_cw_pass", false)},
            {"positive_transfer_official_note",
             confirmatory.value("positive_transfer_official_note", std::string{})},
            {"integration_active", confirmatory.value("integration_active", false)},
            {"gate_f_integrated_v3", confirmatory.value("gate_f_integrated_v3", false)},
            {"gate_f_bounded_v3", confirmatory.value("gate_f_bounded_v3", false)},
            {"gate_f_positive_transfer_track",
             confirmatory.value("gate_f_positive_transfer_track", false)},
            {"gate_e_m2_proxy", confirmatory.value("gate_e_m2_proxy", false)},
            {"gate_f_scenario_c_scientific_strict",
             confirmatory.value("gate_f_scenario_c_scientific_strict", false)},
            {"gate_f_scenario_c_scientific", confirmatory.value("gate_f_scenario_c_scientific", false)},
            {"t5_m1", confirmatory.value("t5_m1", false)},
            {"t5_m2_ran", confirmatory.value("t5_m2_ran", false)},
            {"master_training_checklist_execution_complete",
             confirmatory.value("master_training_checklist_execution_complete", false)},
            {"cells", cells},
        }},
        {"blockers", summary.value("remaining_blockers", json::array())},
        {"python_catalog", catalog_summary},
        {"protocol", parse_protocol_yaml(repo_root / "research_lock" / "FLAGSHIP_DEV_PROTOCOL_v1.yaml")},
        {"claim", build_claim_status(items, summary, confirmatory, gates, flagship_stop)},
    };

    json previous = json::object();
    if (write_history && fs::exists(history_dir)) {
        const fs::path latest = find_latest_history(history_dir, fs::path{});
        if (!latest.empty()) {
            try {
                previous = read_json_file(latest);
            } catch (...) {
            }
        }
    }
    snapshot["diff_vs_previous"] = compute_diff(previous, snapshot);

    json timeline = build_history_timeline(history_dir, 60);
    timeline.push_back(timeline_point(snapshot));
    snapshot["history_timeline"] = timeline;

    fs::path history_archive;
    if (write_history) {
        fs::create_directories(history_dir);
        history_archive = history_dir / ("snapshot_" + history_filename_stamp() + ".json");
        std::ofstream hist_out(history_archive);
        if (hist_out) {
            hist_out << std::setw(2) << snapshot << std::endl;
        }
        prune_history(history_dir, max_history);
    }

    fs::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Failed to write: " << output_path << "\n";
        return 1;
    }
    out << std::setw(2) << snapshot << std::endl;

    std::cout << "Wrote " << output_path << "\n";
    if (write_history && !history_archive.empty()) {
        std::cout << "History: " << history_archive << "\n";
    }
    std::cout << "Timeline points: " << timeline.size() << "\n";
    if (snapshot["diff_vs_previous"].value("has_previous", false)) {
        std::cout << "Checklist delta: "
                  << snapshot["diff_vs_previous"].value("checklist_percent_delta", 0.0)
                  << "%\n";
    }
    std::cout << "Checklist: " << summary.value("done", 0) << "/" << summary.value("total", 0)
              << " (" << summary.value("percent_complete", 0.0) << "%)\n";
    return 0;
}
