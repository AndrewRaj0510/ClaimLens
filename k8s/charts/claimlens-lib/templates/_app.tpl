{{/*
claimlens.app: everything a stateless service needs, driven by its values.
*/}}
{{- define "claimlens.app" -}}
{{ include "claimlens.deployment" . }}
{{- if .Values.service.enabled }}
---
{{ include "claimlens.service" . }}
{{- end }}
{{- if .Values.autoscaling.enabled }}
---
{{ include "claimlens.hpa" . }}
{{- end }}
{{- if .Values.pdb.enabled }}
---
{{ include "claimlens.pdb" . }}
{{- end }}
{{- if include "claimlens.scrape" . }}
---
{{ include "claimlens.serviceMonitor" . }}
{{- end }}
{{- if .Values.networkPolicy.enabled }}
---
{{ include "claimlens.networkPolicy" . }}
{{- end }}
{{- end }}

{{/* Non-empty when Prometheus should scrape this service. */}}
{{- define "claimlens.scrape" -}}
{{- $g := fromYaml (include "claimlens.global" .) -}}
{{- if and .Values.metrics.enabled (dig "serviceMonitors" true $g) -}}yes{{- end -}}
{{- end }}

{{/* The port Prometheus scrapes: the HTTP port, or a separate one (the vision worker has no HTTP API). */}}
{{- define "claimlens.metricsPortName" -}}
{{- if .Values.metrics.port -}}metrics{{- else -}}http{{- end -}}
{{- end }}

{{- define "claimlens.probes" -}}
{{- $p := .Values.probes | default dict -}}
{{- if .Values.healthPath }}
startupProbe:
  {{- if $p.startup }}
  {{- toYaml $p.startup | nindent 2 }}
  {{- else }}
  httpGet: {path: {{ .Values.healthPath }}, port: http}
  periodSeconds: 5
  timeoutSeconds: 3
  # Model loading (guardrails, vision) can take a while; this is the most
  # it may take before liveness starts counting.
  failureThreshold: {{ div (.Values.startupSeconds | default 60) 5 }}
  {{- end }}
readinessProbe:
  {{- if $p.readiness }}
  {{- toYaml $p.readiness | nindent 2 }}
  {{- else }}
  httpGet: {path: {{ .Values.healthPath }}, port: http}
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
  {{- end }}
livenessProbe:
  {{- if $p.liveness }}
  {{- toYaml $p.liveness | nindent 2 }}
  {{- else }}
  httpGet: {path: {{ .Values.healthPath }}, port: http}
  periodSeconds: 20
  timeoutSeconds: 5
  failureThreshold: 3
  {{- end }}
{{- else }}
{{- with $p.startup }}
startupProbe:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $p.readiness }}
readinessProbe:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $p.liveness }}
livenessProbe:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- end }}

{{- define "claimlens.container" -}}
- name: {{ include "claimlens.name" . }}
  image: {{ include "claimlens.image" . }}
  imagePullPolicy: {{ include "claimlens.pullPolicy" . }}
  {{- with .Values.command }}
  command: {{ toJson . }}
  {{- end }}
  {{- with .Values.args }}
  args: {{ toJson . }}
  {{- end }}
  ports:
    {{- if .Values.port }}
    - {name: {{ .Values.portName | default "http" }}, containerPort: {{ .Values.port }}, protocol: TCP}
    {{- end }}
    {{- if .Values.metrics.port }}
    - {name: metrics, containerPort: {{ .Values.metrics.port }}, protocol: TCP}
    {{- end }}
  env:
    {{- include "claimlens.env" . | nindent 4 }}
  {{- include "claimlens.probes" . | nindent 2 }}
  resources:
    {{- toYaml .Values.resources | nindent 4 }}
  securityContext:
    {{- include "claimlens.containerSecurityContext" . | nindent 4 }}
  {{- if or .Values.volumeMounts .Values.mountModels .Values.persistence.enabled }}
  volumeMounts:
    {{- if .Values.mountModels }}
    # Model weights: mounted, never baked into the image (docs/decisions.md 3).
    - {name: models, mountPath: /models, readOnly: true}
    {{- end }}
    {{- if .Values.persistence.enabled }}
    - {name: data, mountPath: {{ .Values.persistence.mountPath }}}
    {{- end }}
    {{- with .Values.volumeMounts }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  {{- end }}
{{- end }}

{{- define "claimlens.podSpec" -}}
# No pod here talks to the Kubernetes API, so none gets a token to do it with.
automountServiceAccountToken: false
{{- with .Values.terminationGracePeriodSeconds }}
terminationGracePeriodSeconds: {{ . }}
{{- end }}
securityContext:
  {{- include "claimlens.podSecurityContext" . | nindent 2 }}
{{- with .Values.imagePullSecrets | default (dig "imagePullSecrets" list (fromYaml (include "claimlens.global" .))) }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
# Spread replicas over nodes where there's more than one (no-op on kind).
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        {{- include "claimlens.selectorLabels" . | nindent 8 }}
containers:
  {{- include "claimlens.container" . | nindent 2 }}
{{- if or .Values.volumes .Values.mountModels }}
volumes:
  {{- if .Values.mountModels }}
  {{- $g := fromYaml (include "claimlens.global" .) }}
  {{- if not $g.modelsVolume }}
  {{- fail "set global.modelsVolume (a volume source): guardrails and the vision workers mount the models from it" }}
  {{- end }}
  - name: models
    {{- toYaml $g.modelsVolume | nindent 4 }}
  {{- end }}
  {{- with .Values.volumes }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
{{- end }}
{{- end }}

{{- define "claimlens.deployment" -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  {{- /* With an autoscaler, it owns the replica count; setting it here would reset it on every upgrade. */}}
  {{- if not (or .Values.autoscaling.enabled .Values.externalScaler) }}
  replicas: {{ .Values.replicas }}
  {{- end }}
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      {{- include "claimlens.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "claimlens.labels" . | nindent 8 }}
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      {{- include "claimlens.podSpec" . | nindent 6 }}
{{- end }}

{{- define "claimlens.service" -}}
apiVersion: v1
kind: Service
metadata:
  # The service name is the chart name, so services find each other as
  # http://gateway:8000 etc., the same names as in docker-compose.yml.
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  type: ClusterIP
  selector:
    {{- include "claimlens.selectorLabels" . | nindent 4 }}
  ports:
    {{- if .Values.port }}
    {{- $portName := .Values.portName | default "http" }}
    - {name: {{ $portName }}, port: {{ .Values.service.port | default .Values.port }}, targetPort: {{ $portName }}, protocol: TCP}
    {{- end }}
    {{- if .Values.metrics.port }}
    - {name: metrics, port: {{ .Values.metrics.port }}, targetPort: metrics, protocol: TCP}
    {{- end }}
{{- end }}

{{- define "claimlens.hpa" -}}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "claimlens.name" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    # Utilisation is relative to the CPU *request*, so requests must be set.
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPU }}
  {{- /* Scale up only on load that lasts a minute: without this, the CPU a
  service burns while it starts (guardrails loading its models) scaled
  guardrails, policy and the frontend to two replicas on kind with no traffic. */}}
  {{- $behavior := dict "scaleUp" (dict "stabilizationWindowSeconds" 60) }}
  behavior:
    {{- toYaml (mergeOverwrite $behavior (.Values.autoscaling.behavior | default dict)) | nindent 4 }}
{{- end }}

{{- define "claimlens.pdb" -}}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  minAvailable: {{ .Values.pdb.minAvailable }}
  selector:
    matchLabels:
      {{- include "claimlens.selectorLabels" . | nindent 6 }}
{{- end }}

{{- define "claimlens.serviceMonitor" -}}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      {{- include "claimlens.selectorLabels" . | nindent 6 }}
  endpoints:
    - port: {{ include "claimlens.metricsPortName" . }}
      path: {{ .Values.metrics.path | default "/metrics" }}
      interval: 15s
{{- end }}

{{/*
Ingress rules for this pod. The umbrella chart denies all ingress in the
namespace by default, so a service is reachable only from the pods and
namespaces listed here, plus Prometheus on the metrics port.
*/}}
{{- define "claimlens.networkPolicy" -}}
{{- $g := fromYaml (include "claimlens.global" .) -}}
{{- $np := .Values.networkPolicy -}}
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  podSelector:
    matchLabels:
      {{- include "claimlens.selectorLabels" . | nindent 6 }}
  policyTypes: [Ingress]
  ingress:
    {{- if and .Values.port (or $np.from $np.fromNamespaces $np.fromPodLabels) }}
    - from:
        {{- range $np.from }}
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: {{ . }}
        {{- end }}
        {{- range $np.fromNamespaces }}
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: {{ . }}
        {{- end }}
        {{- range $np.fromPodLabels }}
        - podSelector:
            matchLabels:
              {{- toYaml . | nindent 14 }}
        {{- end }}
      ports:
        - port: {{ .Values.portName | default "http" }}
    {{- end }}
    {{- if include "claimlens.scrape" . }}
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: {{ $g.monitoringNamespace | default "monitoring" }}
      ports:
        - port: {{ include "claimlens.metricsPortName" . }}
    {{- end }}
    {{- range $np.extraIngress }}
    - {{- toYaml . | nindent 6 }}
    {{- end }}
{{- end }}

{{/*
claimlens.store: a single-replica StatefulSet with its own volume (Postgres,
Redis, SeaweedFS), plus its Service and NetworkPolicy.
*/}}
{{- define "claimlens.store" -}}
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ include "claimlens.name" . }}
  labels:
    {{- include "claimlens.labels" . | nindent 4 }}
spec:
  serviceName: {{ include "claimlens.name" . }}
  replicas: 1
  selector:
    matchLabels:
      {{- include "claimlens.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "claimlens.labels" . | nindent 8 }}
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      {{- include "claimlens.podSpec" . | nindent 6 }}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: [ReadWriteOnce]
        {{- with .Values.persistence.storageClass }}
        storageClassName: {{ . }}
        {{- end }}
        resources:
          requests:
            storage: {{ .Values.persistence.size }}
---
{{ include "claimlens.service" . }}
{{- if .Values.networkPolicy.enabled }}
---
{{ include "claimlens.networkPolicy" . }}
{{- end }}
{{- end }}
