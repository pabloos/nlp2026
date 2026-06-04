# Runbook: Problemas de visibilidad y acceso a datos

**Categoría:** datos  
**Aplica a:** DataVault Pro, CloudSync Enterprise

---

## Usuarios no ven datos del último período (mes, semana)

**Diagnóstico:**
1. ¿El problema afecta a todos los usuarios del tenant o solo a algunos?
2. ¿Los datos existen en el sistema pero no se muestran, o directamente no se ingirieron?
3. Revisar el log de ingesta de datos: Administración → Datos → Historial de sincronización.

**Escenarios:**

### Los datos están en el sistema pero el usuario no los ve
Probablemente es un problema de permisos sobre el período o proyecto:
1. Administración → Permisos de datos → Verificar que el usuario tiene acceso al rango de fechas correspondiente.
2. Algunas organizaciones configuran ventanas de acceso temporales (ej: solo 90 días hacia atrás). Verificar en: Políticas de datos → Retención visible.
3. Refrescar la vista de datos del usuario: panel de Administración → Usuario → Invalidar caché de datos.

### Los datos directamente no están en el sistema
1. Revisar el pipeline de ingesta: ¿el conector de datos está activo?
2. Verificar en el log si hubo errores de sincronización en las fechas afectadas.
3. Si hay un gap de datos (ej: falta todo enero), puede ser un fallo en el job nocturno de sincronización.
4. Escalar a infraestructura con el rango de fechas exacto y el nombre del conector afectado.

---

## Los reportes muestran datos incorrectos o desactualizados

1. Verificar la última sincronización: el timestamp aparece en la esquina inferior del reporte.
2. Si el timestamp es de hace más de 24h, forzar una sincronización manual: Datos → Conectores → Sincronizar ahora.
3. Si los datos son incorrectos (no solo viejos), revisar si las transformaciones del pipeline cambiaron recientemente.
4. Para reportes con agregaciones, verificar que las métricas calculadas no tengan fórmulas desactualizadas.

---

## No se puede acceder a datos después de la actualización del sistema

Después de actualizaciones mayores del sistema, pueden quedar permisos en estado inconsistente:
1. Pedir al usuario que cierre sesión y vuelva a entrar.
2. Si persiste: Administración → Usuario → Sincronizar permisos.
3. En casos extremos, revocar y reasignar los permisos del grupo desde cero.

---

## Escalado

Escalar a datos/infraestructura si:
- Hay un gap de datos mayor a 48h.
- El log de ingesta muestra errores de conexión con la fuente de datos.
- Los datos parecen estar corruptos o mezclados entre tenants (incidente de seguridad potencial — escalar inmediatamente).
