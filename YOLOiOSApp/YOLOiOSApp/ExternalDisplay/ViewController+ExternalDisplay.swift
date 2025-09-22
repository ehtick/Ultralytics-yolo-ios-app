// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// MARK: - OPTIONAL External Display Support
// This extension provides optional external display functionality for the YOLO iOS app.
// It enhances the user experience when connected to an external monitor or TV but is
// NOT required for the core app functionality. The features remain dormant until
// an external display is connected.
//
// Features handled in this extension:
// - External display connection/disconnection detection
// - UI adjustments for external display mode:
//   * Hide switch camera and share buttons (not supported in external display mode)
//   * Adjust model dropdown positioning to prevent overlap
//   * Force landscape orientation for better external display experience
// - Model and threshold synchronization with external display
// - Camera session management (stop iPhone camera when external display is active)

import UIKit
import YOLO

// MARK: - External Display Support
extension ViewController {

  // Associated object key for tracking external display state
  private struct AssociatedKeys {
    static var isExternalDisplayConnected = "isExternalDisplayConnected"
  }

  private var isExternalDisplayConnected: Bool {
    get {
      return objc_getAssociatedObject(self, &AssociatedKeys.isExternalDisplayConnected) as? Bool
        ?? false
    }
    set {
      objc_setAssociatedObject(
        self, &AssociatedKeys.isExternalDisplayConnected, newValue,
        .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
  }

  func setupExternalDisplayNotifications() {
    // Listen for external display connection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayConnected(_:)),
      name: .externalDisplayConnected,
      object: nil
    )

    // Listen for external display disconnection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayDisconnected(_:)),
      name: .externalDisplayDisconnected,
      object: nil
    )

    // Listen for when external display is ready
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayReady(_:)),
      name: .externalDisplayReady,
      object: nil
    )

    // Listen for detection count updates from external display
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleDetectionCountUpdate(_:)),
      name: .detectionCountDidUpdate,
      object: nil
    )
  }

  @objc func handleExternalDisplayConnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.isExternalDisplayConnected = true
      self.yoloView.stop()
      self.yoloView.setInferenceFlag(ok: false)
      self.showExternalDisplayStatus()

      self.requestLandscapeOrientation()

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
        self.adjustLayoutForExternalDisplayIfNeeded()

        self.view.setNeedsLayout()
        self.view.layoutIfNeeded()
        [
          self.yoloView.sliderConf, self.yoloView.labelSliderConf,
          self.yoloView.sliderIoU, self.yoloView.labelSliderIoU,
          self.yoloView.sliderNumItems, self.yoloView.labelSliderNumItems,
          self.yoloView.playButton, self.yoloView.pauseButton,
          self.modelTableView, self.tableViewBGView,
        ].forEach { $0.isHidden = false }

        [
          self.yoloView.switchCameraButton,
          self.yoloView.shareButton,
        ].forEach { $0.isHidden = true }
        self.yoloView.labelSliderNumItems.text =
          "0 items (max \(Int(self.yoloView.sliderNumItems.value)))"

        self.yoloView.sliderNumItems.addTarget(
          self,
          action: #selector(self.updateNumItemsLabelForExternalDisplay),
          for: .valueChanged
        )
        self.modelTableView.setNeedsLayout()
        self.modelTableView.layoutIfNeeded()
        self.tableViewBGView.setNeedsLayout()
        self.tableViewBGView.layoutIfNeeded()
      }

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        if self.currentLoadingEntry != nil || !self.currentModels.isEmpty {
          self.notifyExternalDisplayOfCurrentModel()
        }
        self.sliderValueChanged(self.yoloView.sliderConf)
        NotificationCenter.default.post(
          name: .taskDidChange,
          object: nil,
          userInfo: ["task": self.currentTask]
        )
      }
    }
  }

  private func requestLandscapeOrientation() {
    guard let windowScene = view.window?.windowScene else { return }

    if #available(iOS 16.0, *) {
      windowScene.requestGeometryUpdate(
        .iOS(interfaceOrientations: [.landscapeLeft, .landscapeRight]))
    } else {
      UIViewController.attemptRotationToDeviceOrientation()
    }
  }

  @objc private func updateNumItemsLabelForExternalDisplay() {
    if isExternalDisplayConnected {
      let maxValue = Int(yoloView.sliderNumItems.value)
      let currentText = yoloView.labelSliderNumItems.text ?? ""
      let currentCount = Int(currentText.split(separator: " ").first ?? "0") ?? 0
      yoloView.labelSliderNumItems.text = "\(currentCount) items (max \(maxValue))"
    }
  }

  @objc private func handleDetectionCountUpdate(_ notification: Notification) {
    guard isExternalDisplayConnected,
      let count = notification.userInfo?["count"] as? Int
    else { return }

    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      let maxValue = Int(self.yoloView.sliderNumItems.value)
      self.yoloView.labelSliderNumItems.text = "\(count) items (max \(maxValue))"
    }
  }

  func notifyExternalDisplayOfCurrentModel() {
    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

    var fullModelPath = currentModelName
    if let entry = currentLoadingEntry
      ?? currentModels.first(where: { processString($0.displayName) == currentModelName }),
      entry.isLocalBundle,
      let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
    {
      let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
      fullModelPath = modelURL.path
    }

    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }

  @objc func handleExternalDisplayDisconnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.isExternalDisplayConnected = false
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()

      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      [
        self.yoloView.switchCameraButton,
        self.yoloView.shareButton,
      ].forEach { $0.isHidden = false }

      self.yoloView.sliderNumItems.removeTarget(
        self,
        action: #selector(self.updateNumItemsLabelForExternalDisplay),
        for: .valueChanged
      )
      self.yoloView.sliderChanged(self.yoloView.sliderNumItems)

      self.yoloView.resume()
      self.yoloView.setInferenceFlag(ok: true)

      if let currentEntry = self.currentLoadingEntry {
        if let modelIndex = self.currentModels.firstIndex(where: {
          $0.identifier == currentEntry.identifier
        }) {
          let indexPath = IndexPath(row: modelIndex, section: 0)
          self.modelTableView.selectRow(at: indexPath, animated: false, scrollPosition: .none)
          self.selectedIndexPath = indexPath
          self.loadModel(entry: currentEntry, forTask: self.currentTask)
        } else {
          if !self.currentModels.isEmpty {
            let firstIndex = IndexPath(row: 0, section: 0)
            self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
            self.selectedIndexPath = firstIndex
            self.loadModel(entry: self.currentModels[0], forTask: self.currentTask)
          }
        }
      } else if self.selectedIndexPath == nil && !self.currentModels.isEmpty {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        self.loadModel(entry: self.currentModels[0], forTask: self.currentTask)
      }

      self.requestPortraitOrientation()
    }
  }

  private func requestPortraitOrientation() {
    guard let windowScene = view.window?.windowScene else { return }

    if #available(iOS 16.0, *) {
      windowScene.requestGeometryUpdate(
        .iOS(interfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight]))
      setNeedsUpdateOfSupportedInterfaceOrientations()

      if UIDevice.current.orientation.isLandscape {
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
      }
    } else {
      UIViewController.attemptRotationToDeviceOrientation()

      if UIDevice.current.orientation.isLandscape {
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
      }
    }
  }

  // MARK: - Layout Adjustments for External Display

  func adjustLayoutForExternalDisplayIfNeeded() {
    let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay

    guard hasExternalDisplay else { return }

    let tableViewWidth = view.bounds.width * 0.25

    modelTableView.frame = CGRect(
      x: segmentedControl.frame.maxX + 20,
      y: 20,
      width: tableViewWidth,
      height: 200
    )

    updateTableViewBGFrame()
  }

  @objc func handleExternalDisplayReady(_ notification: Notification) {
    guard !currentTask.isEmpty && !currentModels.isEmpty else { return }

    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

    let currentEntry =
      currentModels.first(where: { processString($0.displayName) == currentModelName })
      ?? currentModels.first
    guard let entry = currentEntry else { return }

    var fullModelPath = ""
    if entry.isLocalBundle,
      let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
    {
      fullModelPath = folderPathURL.appendingPathComponent(entry.identifier).path
    }

    guard !fullModelPath.isEmpty else { return }

    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }

  func checkAndNotifyExternalDisplayIfReady() {
    let hasExternalDisplay = UIApplication.shared.connectedScenes
      .compactMap({ $0 as? UIWindowScene })
      .contains(where: { $0.screen != UIScreen.main })

    if hasExternalDisplay {
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.handleExternalDisplayReady(Notification(name: .externalDisplayReady))
      }
    }
  }

  func checkForExternalDisplays() {
    let hasExternalDisplay = UIScreen.screens.count > 1

    if hasExternalDisplay {
      _ = UIApplication.shared.connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .first(where: { $0.screen != UIScreen.main })
    }
  }

  func showExternalDisplayStatus() {
    let statusLabel = UILabel()
    statusLabel.text = "📱 Camera is shown on external display\n🔄 Please use landscape orientation"
    statusLabel.textColor = .white
    statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.8)
    statusLabel.textAlignment = .center
    statusLabel.font = .systemFont(ofSize: 18, weight: .medium)
    statusLabel.numberOfLines = 0
    statusLabel.adjustsFontSizeToFitWidth = true
    statusLabel.minimumScaleFactor = 0.8
    statusLabel.layer.cornerRadius = 10
    statusLabel.layer.masksToBounds = true
    statusLabel.tag = 9999

    view.addSubview(statusLabel)

    statusLabel.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      statusLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
      statusLabel.widthAnchor.constraint(equalToConstant: 280),
      statusLabel.heightAnchor.constraint(equalToConstant: 140),
    ])
  }

  func hideExternalDisplayStatus() {
    view.subviews.first(where: { $0.tag == 9999 })?.removeFromSuperview()
  }
}
