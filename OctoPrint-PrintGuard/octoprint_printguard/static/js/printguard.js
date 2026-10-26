$(function () {
    function PrintGuardViewModel(parameters) {
        var self = this;
        self.settingsViewModel = parameters[0];
        self.settings = self.settingsViewModel.settings;

        self.isMonitoring = ko.observable(false);
        self.isChecking = ko.observable(false);
        self.activeCamera = ko.observable("camera1");
        self.cameraOptions = ko.observableArray([
            { id: "camera1", label: "Camera 1: USB2.0 PC Camera (/dev/video0)" },
            { id: "camera2", label: "Camera 2: Microsoft LifeCam VX-5000 (/dev/video2)" },
            { id: "phone", label: "Android Phone IP Camera" },
            { id: "both", label: "All Cameras (Camera 1 + Camera 2 + Phone)" },
            { id: "cam1_phone", label: "Camera 1 + Phone Camera (Side-by-Side)" },
            { id: "cam2_phone", label: "Camera 2 + Phone Camera (Side-by-Side)" },
            { id: "custom", label: "Custom Snapshot URL" }
        ]);
        self.checkInterval = ko.observable(10);
        self.intervalOptions = ko.observableArray([
            { val: 5, label: "5 sec" },
            { val: 10, label: "10 sec" },
            { val: 15, label: "15 sec" },
            { val: 30, label: "30 sec" },
            { val: 60, label: "1 min" },
            { val: 120, label: "2 min" },
            { val: 300, label: "5 min" },
            { val: 600, label: "10 min" },
            { val: 1200, label: "20 min" },
            { val: 2400, label: "40 min" },
            { val: 3600, label: "1 hour" }
        ]);
        self.consecutiveFailures = ko.observable(0);
        self.consecutiveThreshold = ko.observable(2);
        self.currentZ = ko.observable(null);
        self.detectionMode = ko.observable("hybrid");
        self.lastStatus = ko.observable("ok");
        self.overallConfidence = ko.observable(0.0);
        self.lastExplanation = ko.observable("Waiting for camera inspection...");
        self.expectedModelSummary = ko.observable("");
        self.scoreNoExtrusion = ko.observable(0.0);
        self.scoreBedDetachment = ko.observable(0.0);
        self.scoreSpaghetti = ko.observable(0.0);
        self.scoreOther = ko.observable(0.0);
        self.history = ko.observableArray([]);
        self.snapshotTimestamp = ko.observable(Date.now());
        self.liveFrameTimestamp = ko.observable(Date.now());
        self.viewFeedMode = ko.observable("annotated");
        self.cam2Connected = ko.observable(false);
        self.isEnlarged = ko.observable(false);
        self.selectedHistoryImageUrl = ko.observable(null);

        self.toggleEnlarge = function () {
            self.selectedHistoryImageUrl(null);
            self.isEnlarged(!self.isEnlarged());
        };

        self.closeEnlarge = function () {
            self.isEnlarged(false);
            self.selectedHistoryImageUrl(null);
        };

        self.openHistoryImage = function (url) {
            if (url) {
                self.selectedHistoryImageUrl(url);
                self.isEnlarged(true);
            }
        };

        self.selectEnlargedFeed = function (mode) {
            self.selectedHistoryImageUrl(null);
            self.viewFeedMode(mode);
        };

        $(document).on("keydown.printguardEnlarge", function (e) {
            if (e.key === "Escape" && self.isEnlarged()) {
                self.closeEnlarge();
            }
        });

        var liveTimer = null;
        self.scheduleNextLiveFrame = function () {
            if (liveTimer) {
                clearTimeout(liveTimer);
                liveTimer = null;
            }
            if (self.viewFeedMode() === "annotated" || self.viewFeedMode() === "gcode") return;
            liveTimer = setTimeout(function () {
                if (self.viewFeedMode() !== "annotated" && self.viewFeedMode() !== "gcode" && !document.hidden) {
                    var tabEl = $("#tab_plugin_printguard");
                    if (!tabEl.length || tabEl.is(":visible")) {
                        self.liveFrameTimestamp(Date.now());
                        return;
                    }
                }
                self.scheduleNextLiveFrame();
            }, 600);
        };

        self.onLiveFrameDone = function () {
            if (self.viewFeedMode() !== "annotated" && self.viewFeedMode() !== "gcode") {
                self.scheduleNextLiveFrame();
            }
        };

        self.viewFeedMode.subscribe(function (newMode) {
            if (newMode !== "annotated" && newMode !== "gcode") {
                self.liveFrameTimestamp(Date.now());
            } else {
                if (newMode === "gcode") {
                    self.snapshotTimestamp(Date.now());
                }
                if (liveTimer) {
                    clearTimeout(liveTimer);
                    liveTimer = null;
                }
            }
        });

        self.snapshotUrl = ko.pureComputed(function () {
            return "plugin/printguard/snapshot.jpg?t=" + self.snapshotTimestamp();
        });

        self.activeFeedUrl = ko.pureComputed(function () {
            var mode = self.viewFeedMode();
            if (mode === "gcode") {
                return "plugin/printguard/expected_model.jpg?t=" + self.snapshotTimestamp();
            }
            if (mode === "phone" || mode === "cam1" || mode === "cam2") {
                return "plugin/printguard/frame/" + mode + ".jpg?t=" + self.liveFrameTimestamp();
            }
            return self.snapshotUrl();
        });

        self.enlargedUrl = ko.pureComputed(function () {
            return self.selectedHistoryImageUrl() || self.activeFeedUrl();
        });

        self.formattedZ = ko.pureComputed(function () {
            var z = self.currentZ();
            return (z !== null && z !== undefined) ? z.toFixed(2) + " mm" : "N/A";
        });

        self.overallConfidencePct = ko.pureComputed(function () {
            return Math.round(self.overallConfidence() * 100) + "%";
        });

        self.scoreNoExtrusionPct = ko.pureComputed(function () {
            return Math.round(self.scoreNoExtrusion() * 100) + "%";
        });

        self.scoreBedDetachmentPct = ko.pureComputed(function () {
            return Math.round(self.scoreBedDetachment() * 100) + "%";
        });

        self.scoreSpaghettiPct = ko.pureComputed(function () {
            return Math.round(self.scoreSpaghetti() * 100) + "%";
        });

        self.scoreOtherPct = ko.pureComputed(function () {
            return Math.round(self.scoreOther() * 100) + "%";
        });

        self.statusBadgeText = ko.pureComputed(function () {
            var st = self.lastStatus();
            if (st === "failure") return "FAILURE DETECTED";
            if (st === "warning") return "WARNING";
            if (self.isMonitoring()) return "MONITORING";
            return "STANDBY";
        });

        self.statusBadgeClass = ko.pureComputed(function () {
            var st = self.lastStatus();
            if (st === "failure") return "label-important";
            if (st === "warning") return "label-warning";
            if (self.isMonitoring()) return "label-success";
            return "label-default";
        });

        self.barClass = function (score) {
            if (score >= 0.70) return "bar-danger";
            if (score >= 0.45) return "bar-warning";
            return "bar-success";
        };

        self.rowBadgeClass = function (status) {
            if (status === "failure") return "label-important";
            if (status === "warning") return "label-warning";
            return "label-success";
        };

        self.formatTimestamp = function (ts) {
            if (!ts) return "-";
            var d = new Date(ts * 1000);
            return d.toLocaleTimeString();
        };

        self.applyState = function (data) {
            if (!data) return;
            self.isMonitoring(!!data.is_monitoring);
            if (data.cam2_connected !== undefined) {
                self.cam2Connected(!!data.cam2_connected);
                if (!data.cam2_connected && self.viewFeedMode() === "cam2") {
                    self.viewFeedMode("phone");
                }
            }
            if (data.camera_options && data.camera_options.length) {
                self.cameraOptions(data.camera_options);
            }
            if (data.active_camera) {
                self.activeCamera(data.active_camera);
            }
            if (data.check_interval_seconds) {
                var iv = parseInt(data.check_interval_seconds, 10);
                var exists = self.intervalOptions().some(function (o) { return o.val === iv; });
                if (!exists) {
                    self.intervalOptions.push({ val: iv, label: iv >= 60 && iv % 60 === 0 ? (iv / 60) + " min" : iv + " sec" });
                }
                self.checkInterval(iv);
            }
            self.consecutiveFailures(data.consecutive_failures || 0);
            self.consecutiveThreshold(data.consecutive_failures_threshold || 2);
            var prevZ = self.currentZ();
            self.currentZ(data.current_z);
            self.detectionMode(data.detection_mode || "hybrid");
            if (data.expected_model_summary !== undefined) {
                self.expectedModelSummary(data.expected_model_summary || "");
            }
            if (data.history) {
                self.history(data.history);
            }
            if (data.last_result) {
                var r = data.last_result;
                self.lastStatus(r.status || "ok");
                self.overallConfidence(r.confidence || 0.0);
                self.lastExplanation(r.explanation || "");
                var sub = r.sub_scores || {};
                self.scoreNoExtrusion(sub.no_extrusion || 0.0);
                self.scoreBedDetachment(sub.bed_detachment || 0.0);
                self.scoreSpaghetti(sub.spaghetti || 0.0);
                self.scoreOther(Math.max(sub.layer_shift || 0.0, sub.nozzle_blob || 0.0));
                self.snapshotTimestamp(Date.now());
            } else if (data.current_z !== prevZ) {
                self.snapshotTimestamp(Date.now());
            }
        };

        self.fetchInitialState = function () {
            OctoPrint.simpleApiGet("printguard").done(function (response) {
                self.applyState(response);
            });
        };

        self.onCameraChange = function () {
            var selected = self.activeCamera();
            self.isChecking(true);
            OctoPrint.simpleApiCommand("printguard", "select_camera", { camera: selected })
                .done(function (response) {
                    self.fetchInitialState();
                })
                .always(function () {
                    self.isChecking(false);
                });
        };

        self.onIntervalChange = function () {
            var iv = parseInt(self.checkInterval(), 10) || 10;
            OctoPrint.simpleApiCommand("printguard", "set_interval", { interval: iv })
                .done(function (response) {
                    if (response && response.state) {
                        self.applyState(response.state);
                    }
                });
        };

        self.toggleMonitoring = function () {
            OctoPrint.simpleApiCommand("printguard", "toggle_monitoring", {})
                .done(function (response) {
                    if (response && response.state) {
                        self.applyState(response.state);
                    }
                });
        };

        self.runManualCheck = function () {
            self.isChecking(true);
            OctoPrint.simpleApiCommand("printguard", "run_check", {})
                .done(function (response) {
                    if (response && response.ok) {
                        self.fetchInitialState();
                    } else if (response && response.error && typeof PNotify !== "undefined") {
                        new PNotify({
                            title: "PrintGuard Check Error",
                            text: response.error,
                            type: "error"
                        });
                    }
                })
                .always(function () {
                    self.isChecking(false);
                });
        };

        self.togglePhoneTorch = function () {
            OctoPrint.simpleApiCommand("printguard", "toggle_phone_torch", {})
                .done(function (response) {
                    if (typeof PNotify !== "undefined") {
                        if (response && response.ok) {
                            new PNotify({
                                title: "Phone Camera Flashlight",
                                text: "Flashlight is now " + (response.torch ? "ON" : "OFF"),
                                type: "info"
                            });
                        } else if (response && response.error) {
                            new PNotify({
                                title: "Phone Torch Error",
                                text: response.error,
                                type: "error"
                            });
                        }
                    }
                });
        };

        self.resetCounter = function () {
            OctoPrint.simpleApiCommand("printguard", "reset_counter", {})
                .done(function (response) {
                    if (response && response.state) {
                        self.applyState(response.state);
                    }
                });
        };

        self.testNotifications = function () {
            OctoPrint.simpleApiCommand("printguard", "test_notification", {})
                .done(function (response) {
                    if (typeof PNotify !== "undefined") {
                        new PNotify({
                            title: "PrintGuard Notification Test",
                            text: "Dispatched test alert to enabled notification channels.",
                            type: "info"
                        });
                    }
                });
        };

        self.onStartupComplete = function () {
            self.fetchInitialState();
        };

        self.onDataUpdaterPluginMessage = function (plugin, data) {
            if (plugin !== "printguard") return;
            var prevStatus = self.lastStatus();
            self.applyState(data);
            if (data.last_result && data.last_result.status === "failure" && prevStatus !== "failure") {
                if (typeof PNotify !== "undefined") {
                    new PNotify({
                        title: "PrintGuard: Print Failure Detected!",
                        text: data.last_result.explanation + (data.last_result.action_taken ? " (" + data.last_result.action_taken + ")" : ""),
                        type: "error",
                        hide: false
                    });
                }
            }
        };
    }

    OCTOPRINT_VIEWMODELS.push({
        construct: PrintGuardViewModel,
        dependencies: ["settingsViewModel"],
        elements: ["#tab_plugin_printguard"]
    });
});
