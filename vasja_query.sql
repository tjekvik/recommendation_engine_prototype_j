select a.appointment_id
	, a.date_sid
	, a.car_vin_number
	, a.car_model as model
	, a.car_brand as brand
	, a.car_vin_number as vin
	, 'service' q_s
	, b.product_catecory category
	, b.service_name headline
from dm.fact_check_appointments a
	join dm.fact_service_added_events b on b.appointment_sid = a.appointment_sid
where a.date_sid > 20250900
union  all
select a.appointment_id
	, a.date_sid
	, a.car_vin_number
	, a.car_model as model
	, a.car_brand as brand
	, a.car_vin_number as vin
	, 'question' q_s	
	, b.product_category
	, b.headline
from dm.fact_check_appointments a
	join dm.fact_question_events b on b.appointment_sid = a.appointment_sid and b.positive = 1
where a.date_sid > 20250900;