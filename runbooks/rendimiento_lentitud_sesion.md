# Runbook: Lentitud de la aplicación y sesiones expiradas

**Categoría:** rendimiento  
**Aplica a:** Todos los productos

---

## Dashboard o reportes cargan muy lento

**Diagnóstico:**
1. Preguntar cuándo empezó el problema — ¿después de una actualización, un lunes por la mañana, siempre?
2. ¿Afecta solo a ese usuario o a toda la organización?
3. Verificar en el panel de estado (status.plataforma.com) si hay incidencias activas.
4. Revisar el volumen de datos del tenant: si supera los 10M de registros, algunos dashboards requieren optimización.

**Resolución por escenario:**

### Lentitud solo en un usuario
1. Pedir al usuario que pruebe en modo incógnito (descarta extensiones del navegador).
2. Verificar la velocidad de conexión del usuario — los dashboards pesados requieren mínimo 10 Mbps.
3. Limpiar la caché de la aplicación: Perfil → Configuración → Limpiar datos de sesión.

### Lentitud generalizada en el tenant
1. Revisar los logs de rendimiento: Administración → Diagnósticos → Tiempo de respuesta por tenant.
2. Si el tiempo de respuesta supera los 5 segundos de media, escalar a infraestructura.
3. Verificar si hay consultas programadas pesadas corriendo en paralelo (reportes automáticos nocturnos que se atrasaron).
4. Como mitigación temporal, deshabilitar los widgets no esenciales del dashboard.

---

## Sesión se cierra automáticamente antes del tiempo configurado

**Configuración por defecto:** 8 horas de inactividad para cuentas corporativas, 1 hora para cuentas básicas.

**Diagnóstico:**
1. Verificar el tiempo de sesión configurado para el tenant: Administración → Seguridad → Políticas de sesión.
2. Comprobar si hay una política de grupo más restrictiva que sobreescribe la configuración del tenant.
3. Revisar si el usuario tiene activo el logout automático en su perfil personal.

**Resolución:**
1. Si la configuración del tenant es correcta, buscar si hay políticas heredadas de una migración anterior.
2. Ajustar el tiempo en: Administración → Seguridad → Duración de sesión → introducir 480 (minutos).
3. Si el problema persiste solo en ciertos navegadores, puede ser un conflicto con las cookies de terceros. Recomendar usar Chrome o Firefox sin modo de privacidad estricta.

---

## Error al exportar datos: timeout o "Exportación fallida"

1. Los exports de más de 100.000 filas pueden tardar hasta 15 minutos. Verificar si el proceso sigue corriendo en: Mis exportaciones → En progreso.
2. Si el export falló, revisar si el usuario tiene el permiso "Exportar datos" activo (ver runbook de permisos).
3. Para exports grandes, usar la exportación asíncrona: el sistema envía un email con el enlace de descarga cuando termina.
4. Si el error es "No autorizado", revisar restricciones de IP (ver runbook de permisos).

---

## Escalado

Escalar a infraestructura si:
- El tiempo de respuesta medio supera 10 segundos durante más de 30 minutos.
- Múltiples tenants reportan lentitud simultáneamente.
- Los logs muestran errores de base de datos (connection timeout, deadlock).
